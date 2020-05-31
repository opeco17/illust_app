CREATE DATABASE generated_imgs;

use generated_imgs;

CREATE TABLE images(
  image_id INT AUTO_INCREMENT PRIMARY KEY,
  image_name VARCHAR(255) not null,
  hair_color VARCHAR(255),
  eye_color VARCHAR(255),
  other_info VARCHAR(255),
  generated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  used_date DATE
) ENGINE=INNODB;


