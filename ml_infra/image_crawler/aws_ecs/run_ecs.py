import json
import uuid

import boto3


def main():
    client = boto3.client('ecs')

    with open('ecs_names.json', 'r') as f:
        names = json.load(f)

    # Create cluster
    client.create_cluster(
        clusterName=names['cluster'],
    )

    # Register task
    client.register_task_definition(
        family=names['task'],
        taskRoleArn='arn:aws:iam::829044821271:role/ImageCrawlerECSTask',
        executionRoleArn='arn:aws:iam::829044821271:role/ImageCrawlerECSTask',
        networkMode='awsvpc',
        containerDefinitions=[
            {
                'name': names['container'],
                'image': '829044821271.dkr.ecr.ap-northeast-1.amazonaws.com/image_crawler:latest',
                'cpu': 4,
                'memory': 1024,
            }
        ]
    )

    # Create and run service
    client.create_service(
        cluster=names['cluster'],
        serviceName=names['service'],
        taskDefinition=names['task'],
        desiredCount=1,
        launchType='EC2',
        networkConfiguration={
            'awsvpcConfiguration': {
                'subnets': [
                    'subnet-07bdef4714c6be1d9',
                ],
                'securityGroups': [
                    'sg-014af1073fc8a5e0f',
                ],
            }
        }
    )



if __name__ == '__main__':
    main()
