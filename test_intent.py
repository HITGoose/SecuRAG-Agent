import sys
from securag_engine import SecuRAG

bot = SecuRAG()

# _decide_intent 内部会吞掉异常并兜底返回 SEARCH，单看返回值分不清"判成了 SEARCH"和"调用失败"。
# 模型不可用时，SEARCH 类用例会全部"答对"，准确率被静默算成 50%。
# 所以这里包一层 LLM 调用，记录每次调用有没有出错。
_original_create = bot.client.chat.completions.create
last_error = None

def tracked_create(*args, **kwargs):
    global last_error
    try:
        return _original_create(*args, **kwargs)
    except Exception as e:
        last_error = e
        raise

bot.client.chat.completions.create = tracked_create

# 预检：先确认模型能调通，不通就直接报错退出，不跑评测
try:
    _original_create(model=bot.model_name, messages=[{"role": "user", "content": "ping"}], max_tokens=1)
except Exception as e:
    print(f"❌ 模型不可用，评测未执行 (mode={bot.mode}, model={bot.model_name})")
    print(f"   错误: {e}")
    print("   本地模式请先 `ollama pull` 对应模型；云端模式请检查 DEEPSEEK_API_KEY。")
    sys.exit(1)

test_cases = [
    {"query": "AMOGEL模型的准确率是多少", "expected": "SEARCH"},
    {"query": "transformer的注意力机制是什么", "expected": "SEARCH"},
    {"query": "RAG和finetune有什么区别", "expected": "SEARCH"},
    {"query": "这篇论文的实验结果如何", "expected": "SEARCH"},
    {"query": "它的参数量是多少", "expected": "SEARCH"},
    {"query": "这个方法的局限性是什么", "expected": "SEARCH"},
    {"query": "论文里提到的数据集叫什么", "expected": "SEARCH"},
    {"query": "baseline模型是什么", "expected": "SEARCH"},
    {"query": "实验用了哪些评估指标", "expected": "SEARCH"},
    {"query": "这个模型在哪个任务上表现最好", "expected": "SEARCH"},
    {"query": "你好", "expected": "CHAT"},
    {"query": "帮我写一段Python冒泡排序", "expected": "CHAT"},
    {"query": "今天天气怎么样", "expected": "CHAT"},
    {"query": "讲个笑话", "expected": "CHAT"},
    {"query": "什么是机器学习", "expected": "CHAT"},
    {"query": "帮我翻译这句话：hello world", "expected": "CHAT"},
    {"query": "你是谁", "expected": "CHAT"},
    {"query": "1+1等于多少", "expected": "CHAT"},
    {"query": "帮我写个邮件模板", "expected": "CHAT"},
    {"query": "推荐几本AI书籍", "expected": "CHAT"},
]

correct = 0
failed = []
for case in test_cases:
    last_error = None
    result = bot._decide_intent(case["query"])
    if last_error is not None:
        failed.append(case["query"])
        print(f"⚠️ [调用失败] {case['query']} -> {last_error}")
        continue
    is_correct = result == case["expected"]
    if is_correct:
        correct += 1
    print(f"{'✅' if is_correct else '❌'} [{case['expected']}→{result}] {case['query']}")

total = len(test_cases)
valid = total - len(failed)
print(f"\n有效用例: {valid}/{total}，调用失败: {len(failed)}")
if failed:
    print(f"❌ 有 {len(failed)} 条用例调用失败，结果无效，不计算准确率")
    sys.exit(1)
print(f"准确率: {correct}/{total} = {correct/total*100:.0f}%")
