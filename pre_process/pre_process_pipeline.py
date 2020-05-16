import logging
import json
import uuid

import boto3
import stepfunctions
from stepfunctions import steps
from stepfunctions.workflow import Workflow


def main():
    stepfunctions.set_stream_logger(level=logging.INFO)
    workflow_execution_role = 'arn:aws:iam::829044821271:role/StepFunctionsWorkflowExecutionRole'

    # Load batch job name
    with open('./face_clip/aws_batch/batch_names.json', 'r') as f:
        face_clip_name = json.load(f)
        
    with open('./tag_extraction/aws_batch/batch_names.json', 'r') as f:
        tag_extraction_name = json.load(f)

    # Define steps
    face_clip_step = steps.BatchSubmitJobStep(
        state_id = 'Face Clip Step',
        parameters={
            'JobDefinition': face_clip_name['jobDefinition'],
            'JobName': face_clip_name['job'],
            'JobQueue': face_clip_name['jobQueue']
        }
    )

    tag_extraction_step = steps.BatchSubmitJobStep(
        state_id = 'Tag Extraction Step',
        parameters={
            'JobDefinition': tag_extraction_name['jobDefinition'],
            'JobName': tag_extraction_name['job'],
            'JobQueue': tag_extraction_name['jobQueue']
        }
    )

    # Define workflow
    chain_list = [face_clip_step, tag_extraction_step]
    workflow_definition = steps.Chain(chain_list)

    workflow = Workflow(
        name='preprocessing_workflow_{0}'.format(uuid.uuid4()),
        definition=workflow_definition,
        role=workflow_execution_role,
    )

    # Execute workflow
    workflow.create()
    workflow.execute()


if __name__ == '__main__':
    main()