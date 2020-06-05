import MySQLdb


class DBConnector(object):
    def __init__(self):
        self.conn = MySQLdb.connect(
            host = 'db', 
            port = 3306,
            user = 'root',
            password = 'root',
            database = 'image_db'
        )
        self.cursor = self.conn.cursor()


    def get_img_name_from_tag(self, hair_color, eye_color, get_img_num):
        self.cursor.execute(
            "SELECT image_name FROM images \
            WHERE hair_color = %s AND eye_color = %s AND used = FALSE \
            LIMIT %s", (hair_color, eye_color, get_img_num)
        )
        result = self.cursor.fetchall()
        result = [img_name[0] for img_name in list(result)]
        return result


    def get_last_img_name(self):
        self.cursor.execute(
            "SELECT image_name FROM images ORDER BY image_id DESC LIMIT 10"
        )
        result = self.cursor.fetchone()
        if result is None:
            return '0.png'
        else:
            return result[0]
        

    def insert_img_info(self, image_name, hair_color, eye_color, other_info=None):
        self.cursor.execute(
            "INSERT INTO images(image_name, hair_color, eye_color, other_info) \
            VALUES (%s, %s, %s, %s)", (image_name, hair_color, eye_color, other_info)
        )


    def stamp_used_img(self, image_name):
        self.cursor.execute(
            "UPDATE images SET used = TRUE WHERE image_name = %s", (image_name,)
        )
    

    def complete(self):
        self.cursor.close()
        self.conn.commit()
        self.conn.close()
        print('Completed!!')
