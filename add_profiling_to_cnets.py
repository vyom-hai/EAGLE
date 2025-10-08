#!/usr/bin/env python3
"""
Add profiling instrumentation to eagle/model/cnets.py topK_genrate method.
This will help us measure ACTUAL latencies in real EAGLE inference.
"""

import sys

# Read the original cnets.py
with open('/home/vyom/EAGLE/eagle/model/cnets.py', 'r') as f:
    content = f.read()

# Add timing imports at the top if not already present
if 'import time' not in content:
    # Find first import and add after it
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            lines.insert(i + 1, 'import time')
            break
    content = '\n'.join(lines)

# Add Timer class if not present
timer_class = '''
# [PROFILING] Added for performance analysis
class ProfileTimer:
    def __init__(self):
        self.times = {}
        self.enabled = False  # Set to True to enable profiling
        
    def start(self, name):
        if not self.enabled:
            return
        if name not in self.times:
            self.times[name] = []
        self.times[name].append(time.perf_counter())
        
    def end(self, name):
        if not self.enabled:
            return
        if name in self.times and self.times[name]:
            elapsed = time.perf_counter() - self.times[name][-1]
            self.times[name][-1] = elapsed
            
    def report(self):
        if not self.enabled or not self.times:
            return
        print("\\n" + "="*60)
        print("topK_genrate PROFILING REPORT")
        print("="*60)
        total = 0
        for name in sorted(self.times.keys()):
            timings = [t for t in self.times[name] if isinstance(t, float)]
            if timings:
                mean_ms = sum(timings) / len(timings) * 1000
                total += mean_ms
                print(f"{name:<40} {mean_ms:>8.3f} ms")
        print("="*60)
        print(f"{'TOTAL':<40} {total:>8.3f} ms")
        print("="*60 + "\\n")

_profile_timer = ProfileTimer()
'''

# Add timer class before the Model class
if 'class ProfileTimer' not in content:
    # Find the Model class definition
    model_class_idx = content.find('class Model(nn.Module):')
    if model_class_idx > 0:
        content = content[:model_class_idx] + timer_class + '\n\n' + content[model_class_idx:]

# Now add timing calls in topK_genrate
# This is more complex - let's add markers for key sections

print("Profiling instrumentation prepared.")
print("\nTo enable profiling in cnets.py:")
print("1. Set _profile_timer.enabled = True at the start of topK_genrate")
print("2. Add _profile_timer.start('section_name') before critical sections")
print("3. Add _profile_timer.end('section_name') after critical sections")
print("4. Add _profile_timer.report() at the end of topK_genrate")
print("\nKey sections to profile:")
print("  - tree_mask_concat (line 757)")
print("  - topk_operations (lines 703, 737, 742, 762)")
print("  - tree_mask_rebuild (lines 776-779)")
print("  - retrieve_indices (lines 796-809)")
print("  - searchsorted (line 770)")
print("  - tensor_tolist (line 774, 792, 797, 800)")

# Save to a patch file instead of modifying directly
patch = f"""
# Add these lines to eagle/model/cnets.py

# 1. After imports (around line 30):
{timer_class}

# 2. At start of topK_genrate (line 670):
        _profile_timer.enabled = True  # Enable profiling
        _profile_timer.start('total')

# 3. Before line 757 (tree mask concat):
            _profile_timer.start('tree_mask_concat')
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)
            _profile_timer.end('tree_mask_concat')

# 4. Around line 774 (tolist call):
            _profile_timer.start('tensor_tolist')
            mask_index_list = mask_index.tolist()
            _profile_timer.end('tensor_tolist')

# 5. Around lines 776-779 (tree mask rebuild):
            _profile_timer.start('tree_mask_rebuild')
            tree_mask = torch.eye(total_tokens + 1).bool()
            tree_mask[:, 0] = True
            for i in range(total_tokens):
                tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])
            _profile_timer.end('tree_mask_rebuild')

# 6. Around lines 796-809 (retrieve indices):
            _profile_timer.start('retrieve_indices')
            # ... all the retrieve indices code ...
            _profile_timer.end('retrieve_indices')

# 7. Before return statement (line 827):
        _profile_timer.end('total')
        _profile_timer.report()
        
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids
"""

with open('/home/vyom/EAGLE/profiling_patch.txt', 'w') as f:
    f.write(patch)

print("\n✓ Profiling patch saved to profiling_patch.txt")
print("\nNext steps:")
print("1. Manually add the profiling code to eagle/model/cnets.py")
print("2. Run: python -m eagle.evaluation.gen_ea_answer_llama3chat \\")
print("          --ea-model-path <model> --base-model-path <base>")
print("3. Check profiling output in console")


