import uuid
import json

import boto3


def main():
    with open('batch_names.json', 'r') as f:
        names = json.load(f)

    client = boto3.client('batch')

    # Submit job
    client.submit_job(
        jobName=names['job']+str(uuid.uuid4()),
        jobQueue=names['jobQueue'],
        jobDefinition=names['jobDefinition'],
    )


if __name__ == '__main__':
    main()
