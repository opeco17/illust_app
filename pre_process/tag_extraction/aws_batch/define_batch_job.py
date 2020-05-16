import time
import json

import boto3


def main():
    with open('batch_names.json', 'r') as f:
        names = json.load(f)

    client = boto3.client('batch')

    # Create compute environment
    print('Start creating compute environment')
    client.create_compute_environment(
        computeEnvironmentName = names['computeEnvironment'],
        type='MANAGED',
        state='ENABLED',
        computeResources = {
            'type': 'EC2',
            'allocationStrategy': 'BEST_FIT',
            'minvCpus': 0,
            'maxvCpus': 16,
            'desiredvCpus': 0,
            'instanceTypes': [
                'optimal'
            ],
            'subnets': [
                'subnet-9e300fc5',
                'subnet-c8d91480',
                'subnet-d6c410fd',
            ],
            'securityGroupIds': [
                'sg-81ec5cf4',
            ],
            'instanceRole': 'arn:aws:iam::829044821271:instance-profile/ecsInstanceRole',
        },
        serviceRole='arn:aws:iam::829044821271:role/AWSBatchServiceRole'
    )

    time.sleep(30)

    # Create job queue
    print('Start creating job queue')
    client.create_job_queue(
        jobQueueName=names['jobQueue'],
        state='ENABLED',
        priority=100,
        computeEnvironmentOrder=[
            {
                'order': 1,
                'computeEnvironment': names['computeEnvironment']
            }
        ]
    )

    time.sleep(30)

    # Register job definition
    print('Start registering job definition')
    client.register_job_definition(
        jobDefinitionName=names['jobDefinition'],
        type='container',
        containerProperties={
            'image': '829044821271.dkr.ecr.ap-northeast-1.amazonaws.com/tag_extraction:latest',
            'vcpus': 4,
            'memory': 4096,
            'jobRoleArn': 'arn:aws:iam::829044821271:role/ImageCrawlerECSTask',
        }
    )


if __name__ == '__main__':
    main()
