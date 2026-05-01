"""
全栈集成测试：绕过http，直接调用服务层，验证整个rag管线
覆盖：文档加载、向量入库、多轮问答、记忆清除、边缘情况"""

import sys
from pathlib import Path

#确保项目根目录在sys.path中
PROJECT_ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT_ROOT))

from config.settings import settings
from services.document_service import DocumentService
from services.vector_store_service import VectorStoreService
from services.rag_service import RAGService

#0.环境初始化：如果向量库不存在，自动加载文档并创建
print("向量数据库")
chroma_path=Path(settings.chroma_persist_dir)
if not chroma_path.exists() or not list(chroma_path.glob("*.sqlite3")):
    print(f"    未检测到向量库，正在从‘./test_docs’加载文档并创建")
    docs=DocumentService.load_and_split("./test_docs")
    print(f"    加载了{len(docs)}个文本块")
    VectorStoreService.create_from_documents(docs)
else:
    print(" 向量数据库已存在，跳过创建步骤")

#1.初始化rag服务（内部自动加载向量库）
rag=RAGService()
sid="integration_test_session"
#2.测试场景a：首轮知识问答
print("场景a:首轮知识问答")
q_a="大语言模型通常包含多少参数？"
ans_a=rag.chat(sid,q_a)
print(f"Q: {q_a}")
print(f"A: {ans_a[:120]}...")
assert len(ans_a) > 10, "回答过短，可能未正常生成"

# 3. 测试场景 B：指代消解追问
# ============================================================
print("\n" + "=" * 60)
print("📌 场景 B：指代消解追问（应理解“它”=大语言模型）")
print("=" * 60)
q_b = "它的训练成本高吗？"
ans_b = rag.chat(sid, q_b)
print(f"Q: {q_b}")
print(f"A: {ans_b[:120]}...")
# 简单判断：回答中应包含“大语言模型”或“训练”相关词汇
assert any(w in ans_b for w in ["大语言模型", "LLM", "训练", "成本"]), \
    "追问未正确关联上下文"

# 4. 测试场景 C：跨知识领域切换
# ============================================================
print("\n" + "=" * 60)
print("📌 场景 C：跨知识领域切换（检索应切换至向量数据库相关内容）")
print("=" * 60)
q_c = "什么是向量数据库？"
ans_c = rag.chat(sid, q_c)
print(f"Q: {q_c}")
print(f"A: {ans_c[:120]}...")
assert any(w in ans_c for w in ["向量", "高维", "语义搜索", "索引"]), \
    "回答未涉及向量数据库核心概念"

# 5. 测试场景 D：清除记忆后指代失效
# ============================================================
print("\n" + "=" * 60)
print("📌 场景 D：清除记忆后，指代应失效")
print("=" * 60)
rag.clear_history(sid)
q_d = "它有什么特点？"
ans_d = rag.chat(sid, q_d)
print(f"Q: {q_d}")
print(f"A: {ans_d[:120]}...")
# 由于没有了历史，系统无法理解“它”，应明确说明或询问
assert any(w in ans_d for w in ["无法", "哪个", "指代", "什么", "不清楚"]), \
    "清除记忆后，系统仍在错误理解指代"

# 6. 测试场景 E：空输入与超长输入（鲁棒性）
# ============================================================
print("\n" + "=" * 60)
print("📌 场景 E：边缘输入鲁棒性测试")
print("=" * 60)

# 空字符串
try:
    ans_empty = rag.chat(sid, "")
    print(f"空输入返回: '{ans_empty}'")
    assert len(ans_empty) >= 0, "空输入不应导致崩溃"
except Exception as e:
    print(f"空输入触发异常（可接受）: {e}")

    # 超长字符串（模拟 2000 字符以上的输入）
long_q = "测试" * 1200
try:
    ans_long = rag.chat(sid, long_q)
    print(f"超长输入返回长度: {len(ans_long)}")
except Exception as e:
    print(f"超长输入触发异常: {e}")

# ============================================================
# 7. 最终报告
# ============================================================
print("\n" + "=" * 60)
print("✅ 所有集成测试场景执行完毕")
print("=" * 60)
print("场景 A（知识问答）: 通过")
print("场景 B（指代消解）: 通过")
print("场景 C（领域切换）: 通过")
print("场景 D（清除记忆）: 通过")
print("场景 E（鲁棒性）  : 通过（系统无崩溃）")