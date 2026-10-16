import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

from app.conf.mineru_config import (
    MineruConfig,
    mineru_config,
    MINERU_ALLOWED_EXTENSIONS,
    MINERU_IMAGE_EXTENSIONS,
)
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import add_running_task, add_done_task

"""
   node_pdf_to_md 
     支持类型：PDF / 图片(png,jpg,jpeg,jp2,webp,gif,bmp) / Office(doc,docx,ppt,pptx,xls,xlsx)
     统一经 MinerU API 转为 Markdown，文件大小上限 200MB、页数上限 200 页
     参数： state [is_pdf_read_enabled = True | pdf_path = xxx.pdf | local_dir = output ]  
     返回： state [md_path = 地址 | md_content = 内容 ]
     1. 日志和任务状态
     2. step_1_validate_paths路径校验
     3. step_2_upload_and_poll minerU的交互
     4. step_3_download_and_extract 下载和解压
     5. 日志和任务状态 return state 
   step_1_validate_paths
     参数：state pdf_path = xxx.pdf | local_dir = output
     返回： pdf_path_obj Path  local_dir_obj Path 
     1. 非空校验 
     2. 文件校验 pdf_path_obj 没有抛异常 local_dir_obj 没有给与默认
     3. 返回完成可用的Path对象即可
   step_2_upload_and_poll
     参数：pdf对应Path  pdf_path_obj
     返回：str zip url地址
     1. 进行申请，获取要上传文件的地址 
     2. 进行文件上传 session | requests.put 
     3. 轮询获取返回结果 zip_url  （确定一个最大等待时间 1页pdf 1s 间隔时间3 错误码 200 -》 500能容忍）
     4. 返回地址即可
   step_3_download_and_extract
     参数：zip_url , out_dir_obj , 原文件名 path.stem
     返回：解压后的.md的str地址 
     1. zip下载 get    output / stem_result.zip
     2. 检查解压的文件夹地址  output / stem 
     3. 检查解压的文件夹进行防重复处理
     4. 进行解压 zipFile  extractall(解压的目标文件夹)
     5. 考虑文件名字 原文件件名 还是 full 还是其他 
     6. 重命名处理
     7. 路径转成字符串 获取绝对路径最终返回即可！
"""



def count_pages(path_obj: Path):
    """
    统计文件页数（尽力而为，无法判定时返回 None，交由 MinerU 服务端兜底校验）。

    - pdf：使用 PyMuPDF 读取真实页数
    - 图片：固定 1 页
    - docx/pptx/xlsx（OOXML 压缩包）：解析内部结构统计页/表/幻灯片数量
    - doc/ppt/xls（旧版二进制）：本地无法解析，返回 None
    :param path_obj: 待解析文件路径
    :return: 页数 int，或 None（无法判定）
    """
    ext = path_obj.suffix.lower()
    if ext == ".pdf":
        import fitz  # PyMuPDF：仅 PDF 需要，延迟导入避免影响其他类型
        with fitz.open(path_obj) as doc:
            return doc.page_count
    if ext in MINERU_IMAGE_EXTENSIONS:
        return 1
    if ext in (".docx", ".pptx", ".xlsx"):
        try:
            with zipfile.ZipFile(path_obj) as zf:
                names = zf.namelist()
                if ext == ".pptx":
                    return len([n for n in names if re.match(r"ppt/slides/slide\d+\.xml$", n)])
                if ext == ".xlsx":
                    return len([n for n in names if re.match(r"xl/worksheets/sheet\d+\.xml$", n)])
                # docx：优先读取 Word 记录的分页数（docProps/app.xml 中的 <Pages>）
                if "docProps/app.xml" in names:
                    app_xml = zf.read("docProps/app.xml").decode("utf-8", "ignore")
                    matched = re.search(r"<Pages>(\d+)</Pages>", app_xml)
                    if matched:
                        return int(matched.group(1))
            return None
        except Exception as e:
            logger.warning(f"[count_pages]解析 {ext} 页数失败，跳过页数校验: {e}")
            return None
    return None


def validate_file_limits(path_obj: Path):
    """
    校验文件大小与页数是否超过 MinerU 上限（默认 大小 200MB / 页数 200 页）。
    :param path_obj: 待解析文件路径
    :return: (file_size_mb, page_count)，page_count 为 None 表示无法判定
    """
    # 1. 大小校验（所有类型均校验）
    max_bytes = mineru_config.max_file_size_mb * 1024 * 1024
    file_size = path_obj.stat().st_size
    if file_size > max_bytes:
        logger.error(f"[validate_file_limits]文件过大: {file_size / 1024 / 1024:.2f}MB > {mineru_config.max_file_size_mb}MB")
        raise ValueError(f"文件大小 {file_size / 1024 / 1024:.2f}MB 超过上限 {mineru_config.max_file_size_mb}MB")
    # 2. 页数校验（可判定类型才校验；旧版二进制 Office 无法判定，交由 MinerU 服务端兜底）
    page_count = count_pages(path_obj)
    if page_count is not None and page_count > mineru_config.max_pages:
        logger.error(f"[validate_file_limits]页数超限: {page_count} > {mineru_config.max_pages}")
        raise ValueError(f"文件页数 {page_count} 超过上限 {mineru_config.max_pages} 页")
    return round(file_size / 1024 / 1024, 2), page_count


def step_1_validate_paths(state):
    """
    步骤1: 参数校验 pdf_path失效则直接异常
                local_dir_obj失效则创建本地目录
    :param state:
    :return:
    """
    logger.debug(f">>> [step_1_validate_paths]开始数据校验")
    pdf_path = state['pdf_path']
    local_dir = state['local_dir']
    # 非空校验
    if not pdf_path:
        logger.error(f"[step_1_validate_paths]参数校验失败: pdf_path 为空")
        raise ValueError("pdf_path 为空")
    if not local_dir:
        local_dir = PROJECT_ROOT / "output"
        logger.info(f"[step_1_validate_paths]参数校验成功: local_dir 创建本地目录 {local_dir}")
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)
    #文件是否存在校验
    if not pdf_path_obj.exists():
        logger.error(f"[step_1_validate_paths]参数校验失败: pdf_path 文件不存在")
        raise FileNotFoundError("pdf_path 文件不存在")
    #文件大小(200MB) / 页数(200页) 上限校验，超限直接异常，避免无效提交给 MinerU
    file_size_mb, page_count = validate_file_limits(pdf_path_obj)
    logger.info(f"[step_1_validate_paths]类型={pdf_path_obj.suffix.lower()} 大小={file_size_mb}MB 页数={page_count}")
    if not local_dir_obj.exists():
        logger.info(f"[step_1_validate_paths]参数校验成功: local_dir 创建本地目录 {local_dir}")
        local_dir_obj.mkdir(parents=True, exist_ok=True)
    return pdf_path_obj, local_dir_obj


def step_2_upload_and_poll(pdf_path_obj) -> str:
    token = mineru_config.api_key
    url = f"{mineru_config.base_url}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": f"{pdf_path_obj.name}"}
        ],
        "model_version": "vlm"
    }
    response = requests.post(url, headers=header, json=data)
    if response.status_code != 200 or response.json().get("code") != 0:
        logger.error(f"[step_2_upload_and_poll]上传文件失败")
        raise RuntimeError("上传文件失败")
    uploaded_url = response.json()['data']['file_urls'][0]
    batch_id = response.json()['data']['batch_id']
    # 用http_session严禁走代理
    http_session = requests.Session()
    http_session.trust_env = False
    # 推荐
    try:
        with open(pdf_path_obj, 'rb') as f:
            file_data = f.read()
        upload_response = http_session.put(uploaded_url,data = file_data)
        if upload_response.status_code != 200:
            logger.error(f"[step_2_upload_and_poll]上传文件失败")
            raise RuntimeError("上传文件失败")
    except Exception as e:
        logger.error(f"[step_2_upload_and_poll]上传文件失败: {e}")
        raise RuntimeError("上传文件失败")
    finally:
        logger.info(f"[step_2_upload_and_poll]上传文件成功")
        http_session.close()

    #获取解析结果
    url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    timeout_seconds = 600
    poll_interval = 3
    start_time = time.time()

    while True:
        #1.超时等待判断
        if time.time() - start_time > timeout_seconds:
            logger.error(f"[step_2_upload_and_poll]获取解析结果超时")
            raise TimeoutError("获取解析结果超时")
        #2.获取解析结果
        res = requests.get(url, headers=header)
        #3.解析结果zip
        if res.status_code != 200:
            if 500 <= res.status_code < 600:
                time.sleep(poll_interval)
                continue
            raise RuntimeError(f"[step_2_upload_and_poll]获取解析结果失败: {res.status_code}")
        json_data = res.json()
        if json_data['code'] != 0:
            raise RuntimeError(f"[step_2_upload_and_poll]获取解析结果失败,返回的错误: {json_data['msg']}")
        extract_result = json_data['data']['extract_result'][0]
        if extract_result['state'] == 'done':
            logger.info(f"[step_2_upload_and_poll]获取解析结果成功")
            full_zip_url = extract_result['full_zip_url']
            return full_zip_url
        else:
            time.sleep(poll_interval)



def step_3_download_and_extract(zip_url, local_dir_obj, stem) -> str:
    """
        下载指定的md.zip文件，并且解压，返回解压后的md文件的地址！
        :param zip_url:  要下载的地址
        :param local_dir_obj: 存储的文件夹
        :param stem: pdf的文件名字
        :return: 返回md文件的地址
        """
    # 1. 下载zip文件 response响应体
    response = requests.get(zip_url)
    if response.status_code != 200:
        logger.error(f"[step_3_download_and_extract]下载zip文件失败")
        raise RuntimeError("下载zip文件失败")
    # 2. 将响应体的zip文件保存到本地
    # 保存文件  output/ 二狗子 / 二狗子_result.zip
    zip_save_path = local_dir_obj / f"{stem}_result.zip"
    with open(zip_save_path, 'wb') as f:
        f.write(response.content)
    logger.info(f"[step_3_download_and_extract]下载zip文件成功,位置: {zip_save_path}")
    # 3. 清空下旧目录（将上一次处理的文件目录进行删除）
    extract_target_dir = local_dir_obj / stem
    # 先清空文件内容，因为两次解压的文件数量可能不一样，会保留旧数据
    if extract_target_dir.exists():
        # 递归进行目录内容删除/本身也会被删除
        shutil.rmtree(extract_target_dir)
    # 创建一个新的目录
    extract_target_dir.mkdir(parents=True, exist_ok=True)
    # 4. 进行zip文件的解压工作
    # python zip解压模块 zipfile进行zip压缩和解压
    # zipfile处理zip文件的模块
    # zipfile.ZipFile(zip_save_path, 'r') as zip_file_object 创建一个zip文件对象，只能读取，解压
    with zipfile.ZipFile(zip_save_path, 'r') as zip_file_object:
        # 解压到指定目录
        zip_file_object.extractall(extract_target_dir)
    # 5. 返回md文件的地址
    # 解压以后得文件的文件名 可能叫 文件.md 低版本 也可能叫 full.md
    md_file_list = list(extract_target_dir.rglob('*.md'))
    if not md_file_list:
        logger.error(f"[step_3_download_and_extract]解压后的md文件不存在")
        raise FileNotFoundError("解压后的md文件不存在")
    target_md_file = None
    # 遍历md文件列表，找到目标md文件
    for md_file in md_file_list:
        if md_file.name == stem + ".md":
            target_md_file = md_file
            break
    #检查full.md文件
    if not target_md_file:
        for md_file in md_file_list:
            if md_file.name.lower()  == "full.md":
                target_md_file = md_file
                break
    #实在没有找到
    if not target_md_file:
        target_md_file = md_file_list[0]

    #如果不是原名则重命名
    if target_md_file.stem != stem:
        #进行重命名
        target_md_file = target_md_file.rename(target_md_file.with_name(f"{stem}.md"))
    #最终路径
    final_md_str_path = str(target_md_file.resolve())
    logger.info(f"[step_3_download_and_extract]解压后的md文件位置: {final_md_str_path}")
    return final_md_str_path







def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    未来要实现:
    1.进入日志参数配置
    2.参数校验
    3.使用minerU解析pdf在线工具
    4.下载zip并解压
    5.赋值md_path,更新state数据状态
    6.结束日志和任务状态配置
    """
    # 1.日志进入节点-》前端
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行节点: {state}")
    add_running_task(state['task_id'], function_name)

    try:
        #2.参数校验
        pdf_path_obj,local_dir_obj = step_1_validate_paths(state)
        #3.使用minerU解析pdf在线工具
        # 参数：pdf_path_obj：pdf文件路径 返回值：zip文件下载链接
        zip_url = step_2_upload_and_poll(pdf_path_obj)
        #4.下载zip并解压
        # 参数：zip_url：zip文件下载链接，local_dir_obj：本地目录，pdf_name：pdf文件名
        # 返回值：md文件路径
        md_path = step_3_download_and_extract(zip_url,local_dir_obj,pdf_path_obj.stem)
        #5.赋值md_path,更新state数据状态
        state['md_path'] = md_path
        state['local_dir'] = str(local_dir_obj)
        #md的内容读取，配置给md_content
        with open(md_path, 'r', encoding='utf-8') as f:
            state['md_content'] = f.read()
    except Exception as e:
        logger.error(f"执行节点: {function_name} 出现异常: {e}")
        raise #终止

    finally:
        logger.info(f">>> [{function_name}]结束执行节点: {state}")
        add_done_task(state['task_id'], function_name)

    return state

if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")