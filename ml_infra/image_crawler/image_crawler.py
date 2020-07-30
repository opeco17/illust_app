import time
import os
import datetime
import json
import logging
from time import sleep

import pandas as pd
import boto3
from pixivpy3 import *


def main():
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('pixiv-image-backet')

    # Logging
    logger = logging.getLogger('Logging')
    fh = logging.FileHandler('logging.log')
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    logger.addHandler(sh)

    # Get bookmarked user id
    bookmark_user_id_df = pd.read_csv('./bookmark_user_id.csv')
    bookmark_user_id_list = list(bookmark_user_id_df.user_id)

    with open('client.json', 'r') as f:
        client_info = json.load(f)

    # Login to pixiv
    api = PixivAPI()
    api.login(client_info["pixiv_id"], client_info["password"])
    aapi = AppPixivAPI()
    aapi.login(client_info["pixiv_id"], client_info["password"])

    # Make directory if not exists
    os.mkdir('./pixiv_images') if not os.path.exists('./pixiv_images') else None

    for user_id in bookmark_user_id_list:
        print(user_id)

        # Get illustrators' information
        illustrator_pixiv_id = user_id
        illustrator_info_json = aapi.user_detail(illustrator_pixiv_id)
        total_illusts = illustrator_info_json.profile.total_illusts
        total_manga = illustrator_info_json.profile.total_manga
        illustrator_name = illustrator_info_json.user.name.replace("/", "-")
        works_info = api.users_works(illustrator_pixiv_id, page=1, per_page=500)

        for i, work_info in enumerate(works_info.response):
            
            # Work painted before 2013 is ignored
            created_date = datetime.datetime.strptime(work_info.created_time, '%Y-%m-%d %H:%M:%S')
            if created_date < datetime.datetime.strptime('2013-1-1 00:00:00', '%Y-%m-%d %H:%M:%S'):
                continue
            
            # Manga is ignored
            if work_info.is_manga:
                continue
            
            # Illustration is downloaded
            work_title = work_info.title.replace('/', '-')
            try:
                aapi.download(work_info.image_urls.large, path='./pixiv_images', name=work_title+str(i)+'.jpg')
                bucket.upload_file(os.path.join('./pixiv_images', work_title+str(i)+'.jpg'), 'raw_images/'+illustrator_name+'/'+work_title+str(i)+'.jpg')
                os.remove(os.path.join('./pixiv_images', work_title+str(i)+'.jpg'))
                print(work_title + 'is downloaded!')
            except Exception as err:
                logger.exception('Raise Exception: %s', err)
                print('Error!')
            sleep(15)


if __name__ == '__main__':
    main()
        
        
        

        


        





