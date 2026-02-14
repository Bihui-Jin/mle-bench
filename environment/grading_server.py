import os
from pathlib import Path

from flask import Flask, jsonify, request

from mlebench.grade import validate_submission, grade_csv
from mlebench.registry import Registry, registry

import logging
# import pandas as pd
# import json

# runs = "/home/b27jin/mle-bench/runs"

# log_list = os.listdir(runs)
# log_list = [f for f in log_list if f.startswith("2024-")]

# df = pd.read_csv(os.path.join(runs, "run_group_experiments.csv"))
# aide = df[df.experiment_id=="scaffolding-gpt4o-aide"]['run_group'].to_list()

# compare = {}
# for dir in aide:
#     log_path = os.path.join(runs, dir)
#     log = os.listdir(log_path)[0]
#     with open(os.path.join(log_path, log), mode='r') as f:
#         report = json.load(f)
    
#     for val in report['competition_reports']:
#         if val['competition_id'] not in compare:
#             compare[val['competition_id']] = val['is_lower_better']


logger = logging.getLogger("aide")
app = Flask(__name__)

PRIVATE_DATA_DIR = "/private/data"
COMPETITION_ID = os.getenv("COMPETITION_ID")  # This is populated for us at container runtime


def run_validation(submission: Path) -> str:
    new_registry = registry.set_data_dir(Path(PRIVATE_DATA_DIR))
    competition = new_registry.get_competition(COMPETITION_ID)
    is_valid, message = validate_submission(submission, competition)
    return message


@app.route("/grade", methods=["POST"])
def grade():
    submission_file = request.files["file"]
    submission_path = Path("/tmp/submission_to_grade.csv")
    submission_file.save(submission_path)


    new_registry = registry.set_data_dir(Path(PRIVATE_DATA_DIR))
    competition = new_registry.get_competition(COMPETITION_ID)
    try:
        report = grade_csv(submission_path, competition)
    except Exception as e:
        logger.info(f'Error occurred while grading: {e}')
        return jsonify({"error": "Grading failed - no report generated"}), 500
    
    logger.info(f'Grading completed successfully with score: {report.score}')

    # is_lower_better: bool = compare.get(COMPETITION_ID, False)
    # if is_lower_better:
    #     report.score = -report.score  # Normalize to higher is better
    # logger.info(f'is_lower_better: {is_lower_better}')

    return jsonify({"score": report.score})

@app.route("/validate", methods=["POST"])
def validate():
    submission_file = request.files["file"]
    submission_path = Path("/tmp/submission_to_validate.csv")
    submission_file.save(submission_path)

    try:
        result = run_validation(submission_path)
    except Exception as e:
        # Server error
        return jsonify({"error": "An unexpected error occurred.", "details": str(e)}), 500

    return jsonify({"result": result})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
