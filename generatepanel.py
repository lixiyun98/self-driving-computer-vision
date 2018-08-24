#载入必要的模块
import pygame
import datetime
import random
from random import choice
import numpy as np
from PIL import Image

#pygame初始化
pygame.init()

# pygame.font.get_fonts()#在测试前首先要知道自己电脑有哪些字体,很多字体不支持中文

chinese = ['浙','苏','沪','京','辽','鲁','闽','陕','渝','川']
number = ['0','1','2','3','5','6','7','8','9']
ALPHABET = ['A','B','C','K','P','S','T','X','Y']
number_lens = 5

#随机生成车牌内容
def random_license_plate_text(chinese=chinese, ALPHABET=ALPHABET, number=number, number_lens=number_lens):

    car_text = []#初始化

    car_text.append(random.choice(chinese))#添加省

    #添加城市
    car_text.append(random.choice(ALPHABET))

    for i in range(number_lens):
        temp = random.choice(number)
        car_text.append(temp)

    return car_text

#将随机产生的车牌内容生成为车牌
def generate_car_license_plate():
    license_plate_text = random_license_plate_text()
    license_plate_text = ''.join(license_plate_text)

    font_styles = ['notosanscjksc', 'notosansschinese','dengxian','fangsong','fzshuti','fzyaoti','kaiti',
                   'lisu','simhei','stfangsong','stkaiti','stsong','stxihei','stxinwei',
                   'stzhongsong','youyuan']

    font_style = choice(font_styles)
    font_size = random.randint(55, 80)
    font = pygame.font.SysFont(font_style, font_size)

    # 渲染图片
    # 第一个参数是写的文字；
    # 第二个参数是个布尔值，以为这是否开启抗锯齿，就是说True的话字体会比较平滑，不过相应的速度有一点点影响；
    # 第三个参数是字体的颜色；
    # 第四个是背景色，如果你想没有背景色（也就是透明），那么可以不加这第四个参数。
    # ftext = font.render(license_plate_text, True, (255, 255, 255),(0, 0, 255))
    ftext = font.render(license_plate_text, False, (255, 255, 255),(60, 71, 188))
    fontWidth = ftext.get_width()
    fontHeight = ftext.get_height()

    img = pygame.image.load("background.jpg")
    img.blit(ftext, ((400-fontWidth) / 2, (100-fontHeight)/2))

    #图片保存到本地
    #nowTime = datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S-%f')
    #pygame.image.save(img, 'Test/'+ nowTime + '.jpg')

    #图片保存到内存
    pil_string_image = pygame.image.tostring(img,"RGB",False)
    line = Image.frombytes("RGB",(400,100),pil_string_image)

    license_plate_image = np.array(line)

    return license_plate_text, license_plate_image

if __name__ == '__main__':

    for i in range(50):
        generate_car_license_plate()
