import glob

# from run import model

class IllustChooser(object):
    @classmethod
    def choose_illust_paths(self):
        illust_paths = glob.glob('./static/*.png')
        return illust_paths[:10]