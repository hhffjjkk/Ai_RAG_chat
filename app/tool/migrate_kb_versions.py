"""
kb_chunks / kb_item_names 集合迁移脚本（版本控制 + 时效性字段上线配套）。

背景：
  版本控制+时效性新增了 kb_chunks 的标量字段（doc_id/version/is_latest/publish_date/
  expire_date/timeliness_status）。Milvus 不支持对已存在集合直接新增字段，因此需要
  drop 旧集合后按新 schema 重建，并重新导入已有文档。

重要说明：
  node_document_split 备份的 chunks.json 属于"切片后、向量化前"的内容，不含向量与
  item_name，无法直接回填向量。因此本脚本采用"重新导入原始文件"的方式重建数据：
  遍历 output/{日期}/{task_id}/ 下的原始文件（.pdf/.md），重新执行导入图。

用法（在项目根目录执行）：
  # 1) 仅预览将重建的集合与可重导的文件（不修改数据）
  python -m app.tool.migrate_kb_versions

  # 2) 确认后执行：drop 旧集合 + 重建 schema + 重新导入
  python -m app.tool.migrate_kb_versions --drop --reimport

  # 3) 仅重建 schema，不重导（自行批量重导）
  python -m app.tool.migrate_kb_versions --drop

注意：执行 --drop 会清空现有 kb_chunks / kb_item_names 数据，请先确认已有备份。
"""
import argparse
import uuid

from app.core.logger import logger
from app.conf.milvus_config import milvus_config
from app.utils.path_util import PROJECT_ROOT
from app.clients.milvus_utils import get_milvus_client

# 可重新导入的源文件后缀
_SOURCE_SUFFIXES = (".pdf", ".md")


def _iter_source_files():
    """
    遍历可重新导入的"原始文件"。
    原始文件位于 output/{日期}/{task_id}/{文件名}（相对 output 深度为 3），
    而 mineru 产出的中间文件位于 output/{日期}/{task_id}/{stem}/（深度为 4），
    因此仅保留相对 output 深度为 3 的 pdf/md 文件，避免重复导入。
    """
    output_dir = PROJECT_ROOT / "output"
    if not output_dir.exists():
        return []
    files = []
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _SOURCE_SUFFIXES:
            continue
        # 跳过 md_img 节点产出的中间文件
        if path.name.endswith("_new.md"):
            continue
        # 仅保留 output/{日期}/{task_id}/{文件} 这一层的原始文件
        rel = path.relative_to(output_dir)
        if len(rel.parts) != 3:
            continue
        files.append(path)
    return files


def _drop_and_recreate(milvus_client):
    """删除旧集合（kb_chunks / kb_item_names）"""
    for collection in (milvus_config.chunks_collection, milvus_config.item_name_collection):
        if milvus_client.has_collection(collection_name=collection):
            milvus_client.drop_collection(collection_name=collection)
            logger.info(f"已删除集合：{collection}")
        else:
            logger.info(f"集合不存在，跳过删除：{collection}")


def _recreate_chunks():
    """按新 schema 重建 kb_chunks（复用导入节点的建表逻辑，保证字段一致）"""
    from app.import_process.agent.nodes.node_import_milvus import step_2_prepare_collections
    step_2_prepare_collections({})
    logger.info(f"已按新 schema 重建集合：{milvus_config.chunks_collection}")


def _reimport_files():
    """重新执行导入图，重建切片/向量/版本注册信息"""
    from app.import_process.agent.main_graph import kb_import_app
    from app.import_process.agent.state import get_default_state

    files = _iter_source_files()
    logger.info(f"待重新导入的原始文件数量：{len(files)}")
    for file_path in files:
        task_id = str(uuid.uuid4())
        state = get_default_state()
        state["task_id"] = task_id
        state["local_file_path"] = str(file_path)
        state["local_dir"] = str(file_path.parent)
        try:
            for _ in kb_import_app.stream(state):
                pass
            logger.info(f"重新导入成功：{file_path}")
        except Exception as e:
            logger.error(f"重新导入失败：{file_path}，原因：{e}")
    logger.info("kb_item_names 集合将在重新导入过程中按需创建/写入")


def main():
    parser = argparse.ArgumentParser(description="kb_chunks / kb_item_names 迁移与重建脚本")
    parser.add_argument("--drop", action="store_true", help="删除并重建集合（会清空现有数据）")
    parser.add_argument("--reimport", action="store_true", help="重建后重新导入 output 下的原始文件")
    args = parser.parse_args()

    milvus_client = get_milvus_client()
    if milvus_client is None:
        logger.error("无法连接 Milvus，请检查 MILVUS_URL 配置与服务状态")
        return

    files = _iter_source_files()
    logger.info("=" * 60)
    logger.info(f"kb_chunks 集合：{milvus_config.chunks_collection}")
    logger.info(f"kb_item_names 集合：{milvus_config.item_name_collection}")
    logger.info(f"可重新导入的原始文件数量：{len(files)}")
    for f in files[:20]:
        logger.info(f"  - {f}")
    if len(files) > 20:
        logger.info(f"  ... 其余 {len(files) - 20} 个文件省略")
    logger.info("=" * 60)

    if not args.drop:
        logger.info("未传入 --drop，仅预览。确认后请加上 --drop 执行迁移。")
        return

    _drop_and_recreate(milvus_client)
    _recreate_chunks()

    if args.reimport:
        _reimport_files()
    else:
        logger.info("未传入 --reimport：已重建空集合，请自行重新导入历史文档。")
    logger.info("迁移脚本执行完毕")


if __name__ == "__main__":
    main()
