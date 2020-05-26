import uuid
import logging

import boto3
from boto3 import Session
import stepfunctions
import sagemaker
from stepfunctions.template.pipeline import TrainingPipeline
from stepfunctions import steps
from stepfunctions.workflow import Workflow
from sagemaker.pytorch import PyTorch



def main():
    sagemaker_session = sagemaker.Session()
    stepfunctions.set_stream_logger(level=logging.INFO)

    bucket = 's3://pixiv-image-backet'

    sagemaker_execution_role = 'arn:aws:iam::829044821271:role/service-role/AmazonSageMaker-ExecutionRole-20200412T194702'
    workflow_execution_role = 'arn:aws:iam::829044821271:role/StepFunctionsWorkflowExecutionRole'

    estimator1 = PyTorch(entry_point='train.py', 
                        source_dir='projection_discriminator',
	                    role=sagemaker_execution_role,
                        framework_version='1.4.0',
                        train_instance_count=2,
                        train_instance_type='ml.m5.2xlarge',
                        hyperparameters={
                            'train_epoch' : 1,
                        }
                    )

    estimator2 = PyTorch(entry_point='train.py', 
                        source_dir='wgan_gp',
    	                role=sagemaker_execution_role,
                        framework_version='1.4.0',
                        train_instance_count=2,
                        train_instance_type='ml.m5.2xlarge',
                        hyperparameters={
                            'train_epoch' : 1,
                        }
                    )

    training_step1 = steps.TrainingStep(
        state_id='Train Step1', 
        estimator=estimator1,
        data={
            'training': bucket,
        },
        job_name='PD-Train-{0}'.format(uuid.uuid4())
    )

    training_step2 = steps.TrainingStep(
        state_id='Train Step2', 
        estimator=estimator2,
        data={
            'training': bucket,
        },
        job_name='PD-Train-{0}'.format(uuid.uuid4())
    )

    parallel_state = steps.Parallel(
        state_id='Parallel',
    )

    parallel_state.add_branch(training_step1)
    parallel_state.add_branch(training_step2)

    workflow_definition = steps.Chain([parallel_state])

    workflow = Workflow(
        name='MyTraining-{0}'.format(uuid.uuid4()),
        definition=workflow_definition,
        role=workflow_execution_role,
    )

    workflow.create()
    workflow.execute()


if __name__ == '__main__':
    main()
