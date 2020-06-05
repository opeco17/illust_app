import json

import boto3


def main():
    client = boto3.client('events')

    with open('./stepfunctions_name.json', 'r') as f:
        names = json.load(f)

    client.put_rule(
        Name=names['cloudwatch_events_name'],
        ScheduleExpression='cron(0 3 * * ? *)',
        State='ENABLED',
    )

    client.put_targets(
        Rule=names['cloudwatch_events_name'],
        Targets=[
            {
                'Id': names['cloudwatch_events_id'],
                'Arn': 'arn:aws:states:ap-northeast-1:829044821271:stateMachine:preprocess_pipeline',      
                'RoleArn': 'arn:aws:iam::829044821271:role/service-role/AWS_Events_Invoke_Step_Functions_1710309540'
            }
        ]
    )


if __name__ == '__main__':
    main()
