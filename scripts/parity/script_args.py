"""Parse the run_benchmark.py invocations of TAB's benchmark scripts."""
import glob
import json
import os
import shlex


def parse_script_line(line):
    """Return the arguments of one ``run_benchmark.py`` command line as a dict."""
    tokens = shlex.split(line)
    args = {}
    for i, tok in enumerate(tokens):
        if tok.startswith("--") and i + 1 < len(tokens):
            args[tok[2:]] = tokens[i + 1]
    return {
        "config_path": args.get("config-path"),
        "model_name": args.get("model-name"),
        "adapter": args.get("adapter"),
        "hyper_params": json.loads(args.get("model-hyper-params", "{}")),
    }


def parse_script(path):
    """Yield the parsed invocations of one script file."""
    with open(path) as f:
        for line in f:
            if "run_benchmark.py" in line:
                entry = parse_script_line(line)
                entry["script"] = path
                yield entry


def parse_scripts(root="scripts"):
    """Yield one entry per distinct run_benchmark.py invocation under ``root``."""
    seen = set()
    for path in sorted(glob.glob(os.path.join(root, "**", "*.sh"), recursive=True)):
        for entry in parse_script(path):
            key = (
                entry["config_path"],
                entry["model_name"],
                entry["adapter"],
                json.dumps(entry["hyper_params"], sort_keys=True),
            )
            if key not in seen:
                seen.add(key)
                yield entry
