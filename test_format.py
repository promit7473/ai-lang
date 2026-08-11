import sys
sys.path.insert(0, "src")
from ai_lang import parse, compile_task, run
from ai_lang.llm import from_env

llm = from_env()
source = open("examples/vessel_monitor.ais").read()
task = parse(source)
graph = compile_task(task)

print("=== COMPACT SCHEMA + REPAIR LOOP ===")
for retries in [1, 2, 3]:
    result = run(graph, inputs={"camera": "cam_01", "target_id": "vessel_42"}, llm=llm, use_llm=True, max_retries=retries)
    print(f"Retries={retries}: calls={result.total_llm_calls}, tokens={result.total_llm_tokens}, cost=${result.total_llm_cost:.6f}, repairs={result.repair_attempts}, errors={len(result.errors)}")

print()
print("=== NATURAL LANGUAGE ESTIMATE ===")
nl_total = 2500 + (800 * 6)
print(f"Estimated: ~{nl_total} tokens, ${nl_total * 0.0000025:.6f}")
