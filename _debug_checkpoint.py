import inspect

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

out = []
out.append("=== BaseCheckpointSaver.list ===")
out.append(inspect.getsource(BaseCheckpointSaver.list))
out.append("=== BaseCheckpointSaver.put_writes ===")
out.append(inspect.getsource(BaseCheckpointSaver.put_writes))
out.append("=== BaseCheckpointSaver.get_next_version ===")
out.append(inspect.getsource(BaseCheckpointSaver.get_next_version))
out.append("=== InMemorySaver.put ===")
out.append(inspect.getsource(InMemorySaver.put))
out.append("=== InMemorySaver.get_tuple ===")
out.append(inspect.getsource(InMemorySaver.get_tuple))
out.append("=== InMemorySaver.list ===")
out.append(inspect.getsource(InMemorySaver.list))

with open("_debug_checkpoint_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("DONE")
