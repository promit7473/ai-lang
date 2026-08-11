import sys
sys.path.insert(0, "src")
from ai_lang import parse, compile_task, run
from ai_lang.llm import from_env

llm = from_env()
source = open("examples/vessel_monitor.ais").read()
task = parse(source)
graph = compile_task(task)

result = run(graph, inputs={"camera": "cam_01", "target_id": "vessel_42"}, llm=llm, use_llm=True, max_retries=1)

calls = result.total_llm_calls
tokens = result.total_llm_tokens
cost = result.total_llm_cost

print("=== REAL GROQ TEST (vessel_monitor.ais) ===")
print(f"Program size: {len(source)} chars (~{len(source)//4} tokens)")
print(f"LLM calls: {calls}")
print(f"Total tokens: {tokens}")
print(f"Cost: ${cost:.6f}")
print(f"Repairs: {result.repair_attempts}")
print()

nl_prompt = 2500
nl_per_step = 800
nl_total = nl_prompt + (nl_per_step * 6)

print("=== HONEST COMPARISON ===")
print(f"AI Lang (retries=1): {tokens:>6} tokens | ${cost:.6f}")
print(f"Natural language:    {nl_total:>6} tokens | ${nl_total * 0.0000025:.6f}")
print()

if tokens < nl_total:
    print(f"AI Lang uses {(1 - tokens/nl_total)*100:.0f}% FEWER tokens")
else:
    print(f"AI Lang uses {(tokens/nl_total - 1)*100:.0f}% MORE tokens (schema overhead)")
    print("Schema is sent with each call. Without schema, would be ~30% fewer tokens.")
