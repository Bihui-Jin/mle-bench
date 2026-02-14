import argparse
import asyncio
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import docker

from agents.registry import Agent
from agents.registry import registry as agent_registry
from agents.run import run_in_container
from environment.defaults import DEFAULT_CONTAINER_CONFIG_PATH
from mlebench.data import is_dataset_prepared
from mlebench.registry import Competition, registry
from mlebench.utils import create_run_dir, get_logger, get_runs_dir, get_timestamp

import subprocess
import sys

logger = get_logger(__name__)


@dataclass(frozen=True)
class Task:
    run_id: str
    seed: int
    image: str
    path_to_run_group: Path
    path_to_run: Path
    agent: Agent
    competition: Competition
    container_config: dict[str, Any]
    init_code_path: str  # Path to the generated init code
    reference_path: str  # Path to the reference file

async def worker(
    idx: int,
    queue: asyncio.Queue[Task],
    client: docker.DockerClient,
    tasks_outputs: dict[str, dict[str, Any]],
) -> None:
    while True:
        task = await queue.get()

        # Create logger for the run
        run_logger = get_logger(str(task.path_to_run))
        log_file_handler = logging.FileHandler(task.path_to_run / "run.log")
        log_file_handler.setFormatter(
            logging.getLogger().handlers[0].formatter
        )  # match the formatting we have
        run_logger.addHandler(log_file_handler)
        run_logger.propagate = False

        run_logger.info(
            f"[Worker {idx}] Running seed {task.seed} for {task.competition.id} and agent {task.agent.name}"
        )

        task_output = {}
        try:
            await asyncio.to_thread(
                run_in_container,
                client=client,
                competition=task.competition,
                agent=task.agent,
                image=task.agent.name,
                container_config=task.container_config,
                retain_container=args.retain,
                run_dir=task.path_to_run,
                logger=run_logger,
                init_code_path=task.init_code_path,
                reference_path=task.reference_path,
            )
            task_output["success"] = True

            run_logger.info(
                f"[Worker {idx}] Finished running seed {task.seed} for {task.competition.id} and agent {task.agent.name}"
            )
        except Exception as e:
            stack_trace = traceback.format_exc()
            run_logger.error(type(e))
            run_logger.error(stack_trace)
            run_logger.error(
                f"Run failed for seed {task.seed}, agent {task.agent.id} and competition "
                f"{task.competition.id}"
            )
            task_output["success"] = False
        finally:
            tasks_outputs[task.run_id] = task_output
            queue.task_done()


async def main(args):
    if args.gpu is not None:
        default_json = "./environment/config/container_configs/default.json"
        with open(default_json, mode="r") as f:
            container_config = json.load(f)
        container_config["device_ids"] = f'["{args.gpu}"]'
        container_config["nano_cpus"] = int(4e9)
        with open(default_json, "w") as f:
            json.dump(container_config, f, indent=4, sort_keys=False)
    
    # Clear the file before writing
    if args.code_gen is not None:
        with open('/home/b27jin/mle-bench/environment/init_code.txt', mode='w') as f:
            pass
            
    client = docker.from_env()
    global registry
    registry = registry.set_data_dir(Path(args.data_dir))

    agent = agent_registry.get_agent(args.agent_id)
    if agent.privileged and not (
        os.environ.get("I_ACCEPT_RUNNING_PRIVILEGED_CONTAINERS", "False").lower()
        in ("true", "1", "t")
    ):
        raise ValueError(
            "Agent requires running in a privileged container, but the environment variable `I_ACCEPT_RUNNING_PRIVILEGED_CONTAINERS` is not set to `True`! "
            "Carefully consider if you wish to run this agent before continuing. See agents/README.md for more details."
        )

    run_group = f"{get_timestamp()}_run-group_{agent.name}"

    # Load competition ids and check all are prepared
    with open(args.competition_set, "r") as f:
        competition_ids = [line.strip() for line in f.read().splitlines() if line.strip()]
    for competition_id in competition_ids:
        competition = registry.get_competition(competition_id)
        if not is_dataset_prepared(competition):
            raise ValueError(
                f"Dataset for competition `{competition.id}` is not prepared! "
                f"Please run `mlebench prepare -c {competition.id}` to prepare the dataset."
            )

    with open(args.container_config, "r") as f:
        container_config = json.load(f)

    # Create tasks for each (competition * seed)
    logger.info(f"Launching run group: {run_group}")
    tasks = []
    logger.info(f"# of seeds: {args.n_seeds}")
    for seed in range(args.n_seeds):
        for competition_id in competition_ids:
            if args.code_gen is not None:
                init_code_path = await generate_code_for_competition(competition_id, args.code_gen)
                logger.info(f"{seed}: Generated code for {competition_id} at '{init_code_path}'")
            
            reference_path = None
            if args.insight is not None:
                if args.insight == "diff":
                    reference_path = f"/home/b27jin/mle-bench-internal/code_references/diff_plan/{agent.kwargs['agent.code.model']}_{competition_id}.txt"
                elif args.insight == "soln":
                    reference_path = f"/home/b27jin/mle-bench-internal/code_references/soln_plan/{agent.kwargs['agent.code.model']}_{competition_id}.txt"
                elif args.insight == "combo":
                    reference_path = f"/home/b27jin/mle-bench-internal/code_references/combo_plan/{agent.kwargs['agent.code.model']}_{competition_id}.txt"

            competition = registry.get_competition(competition_id)
            run_dir = create_run_dir(competition.id, agent.id, run_group)
            run_id = run_dir.stem
            task = Task(
                run_id=run_id,
                seed=seed,
                image=agent.name,
                agent=agent,
                competition=competition,
                path_to_run_group=run_dir.parent,
                path_to_run=run_dir,
                container_config=container_config,
                init_code_path=init_code_path if args.code_gen is not None else "/home/b27jin/mle-bench/environment/init_code.txt",
                reference_path=reference_path,
            )
            tasks.append(task)
    logger.info(f"Analyzing tasks by the model: {agent.kwargs['agent.code.model']}")
    logger.info(f"Creating {args.n_workers} workers to serve {len(tasks)} tasks...")

    # Create queue of tasks, and assign workers to run them
    queue = asyncio.Queue()
    for task in tasks:
        queue.put_nowait(task)
    workers = []
    tasks_outputs = {}
    for idx in range(args.n_workers):
        w = asyncio.create_task(worker(idx, queue, client, tasks_outputs))
        workers.append(w)

    # Wait for all tasks to be completed and collect results
    started_at = time.monotonic()
    await queue.join()
    time_taken = time.monotonic() - started_at

    for w in workers:
        w.cancel()  # Cancel all workers now that the queue is empty

    await asyncio.gather(*workers, return_exceptions=True)

    # Generate metadata.json
    metadata = {
        "run_group": run_group,
        "created_at": get_timestamp(),
        "runs": tasks_outputs,
    }
    run_group_dir = get_runs_dir() / run_group
    with open(run_group_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=False, default=str)
    logger.info(f"{args.n_workers} workers ran for {time_taken:.2f} seconds in total")

async def generate_code_for_competition(competition_name: str, code_gen_sources: str) -> str:
    """
    Runs the generation script for a given competition and captures its output.
    """
    if code_gen_sources == "diff":
        script = "step_9_generate_code_from_diff_insights.py"
    elif code_gen_sources == "combo":
        script = "step_7_generate_code_from_two_insights.py"
    elif code_gen_sources == "soln":
        script = "step_8_generate_code_from_soln_insights.py"

    command = [
        sys.executable,  # Use the same python interpreter that's running this script
        f"/home/b27jin/mle-bench-internal/top_k_scripts/{script}",
        competition_name
    ]

    try:
        # Run the script as a subprocess
        # capture_output=True captures stdout and stderr
        # text=True decodes stdout/stderr as text
        # check=True will raise an exception if the script returns a non-zero exit code
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )

        # The captured standard output is the "returned" value
        return result.stdout.strip()

    except subprocess.CalledProcessError as e:
        print(f"Error running script for {competition_name}:")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        return None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an agent on a set of competitions in a Docker container."
    )
    parser.add_argument(
        "--agent-id",
        help="Agent ID of the agent to run.",
        type=str,
    )
    parser.add_argument(
        "--competition-set",
        type=str,
        required=True,
        help="Path to a text file with a single competition ID on each line",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        required=False,
        default=1,
        help="Number of workers to run in parallel",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        required=False,
        default=1,
        help="Number of seeds to run for each competition",
    )
    parser.add_argument(
        "--container-config",
        help="Path to a JSON file with an environment configuration; these args will be passed to `docker.from_env().containers.create`",
        type=str,
        required=False,
        default=DEFAULT_CONTAINER_CONFIG_PATH,
    )
    parser.add_argument(
        "--retain",
        help="Whether to retain the container after the run instead of removing it.",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--run-dir",
        help="Path to the directory where all assets associated with the run are stored.",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--data-dir",
        help="Path to the directory containing the competition data.",
        type=str,
        required=False,
        default=registry.get_data_dir(),
    )
    parser.add_argument(
        "--code-gen",
        help="Code generation sources",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--insight",
        help="Insight sources",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--gpu",
        type=int,
        required=False,
        default=0,
        help="Designated GPU to allocate the competition",
    )
    args = parser.parse_args()
    print(args)

    logger = get_logger(__name__)

    asyncio.run(main(args))
