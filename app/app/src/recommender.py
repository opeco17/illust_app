import random
import glob

# from run import model

class IllustChooser(object):
    @classmethod
    def choose_illust_paths(self):
        illust_paths = glob.glob('./static/*.png')
        illust_paths.remove('./static/sample_img.png')
        random.shuffle(illust_paths)
        return illust_paths[:10]
