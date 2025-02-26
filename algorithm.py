import os
import cv2
import json
import numpy as np
from numpy.linalg import norm

SZ = 20                    # 训练图片长宽
MAX_WIDTH = 1000           # 原始图片最大宽度
Min_Area = 2000            # 车牌区域允许最大面积
PROVINCE_START = 1000      # 省份起始标识符


# 读取图片文件
def imreadex(filename):
	return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)


# 限制点坐标在图片范围内
def point_limit(point):
	if point[0] < 0:
		point[0] = 0
	if point[1] < 0:
		point[1] = 0


# 根据设定的阈值和图片直方图，找出波峰，用于分隔字符
def find_waves(threshold, histogram):
	up_point = -1  # 上升点
	is_peak = False
	if histogram[0] > threshold:
		up_point = 0
		is_peak = True
	wave_peaks = []
	for i, x in enumerate(histogram):
		if is_peak and x < threshold:
			if i - up_point > 2:
				is_peak = False
				wave_peaks.append((up_point, i))
		elif not is_peak and x >= threshold:
			is_peak = True
			up_point = i
	if is_peak and up_point != -1 and i - up_point > 4:
		wave_peaks.append((up_point, i))
	return wave_peaks


# 根据找出的波峰，分隔图片，从而得到逐个字符图片
def seperate_card(img, waves):
	part_cards = []
	for wave in waves:
		part_cards.append(img[:, wave[0]:wave[1]])
	return part_cards


# 矫正图像倾斜
def deskew(img):
	m = cv2.moments(img)
	if abs(m['mu02']) < 1e-2:
		return img.copy()
	skew = m['mu11']/m['mu02']
	m_ = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
	img = cv2.warpAffine(img, m_, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
	return img


# 字符模板
template = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
			'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M',
			'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W','X', 'Y', 'Z',
			'藏', '川', '鄂', '甘', '赣', '贵', '桂', '黑', '沪', '吉', '冀',
			'津', '晋', '京', '辽', '鲁', '蒙', '闽', '宁', '青', '琼', '陕',
			'苏', '皖', '湘', '新', '渝', '豫', '粤', '云', '浙']


# 读取一个文件夹下的所有图片，输入参数是文件名，返回模板文件地址列表
def read_directory(directory_name):
	referImg_list = []
	for filename in os.listdir(directory_name):
		referImg_list.append(directory_name + "/" + filename)
	return referImg_list


# 获得中文模板列表（只匹配车牌的第一个字符）
def get_chinese_words_list():
	chinese_words_list = []
	for i in range(34, 64):
		# 将模板存放在字典中
		c_word = read_directory('./template/' + template[i])
		chinese_words_list.append(c_word)
	return chinese_words_list


# 获得英文模板列表（只匹配车牌的第二个字符）
def get_eng_words_list():
	eng_words_list = []
	for i in range(10, 34):
		e_word = read_directory('./template/' + template[i])
		eng_words_list.append(e_word)
	return eng_words_list


# 获得英文和数字模板列表（匹配车牌后面的字符）
def get_eng_num_words_list():
	eng_num_words_list = []
	for i in range(0, 34):
		word = read_directory('./template/' + template[i])
		eng_num_words_list.append(word)
	return eng_num_words_list


#  车牌图像预处理
def car_pic_preprocess(car_pic, resize_rate):
	if isinstance(car_pic, str):
		img = imreadex(car_pic)
	else:
		img = car_pic
	pic_height, pic_width = img.shape[:2]

	# 如果图片宽度超过最大宽度限制，进行缩放
	if pic_width > MAX_WIDTH:
		pic_rate = MAX_WIDTH / pic_width
		img = cv2.resize(img, (MAX_WIDTH, int(pic_height * pic_rate)), interpolation=cv2.INTER_LANCZOS4)
		pic_height, pic_width = img.shape[:2]

	# 如果设置了缩放比例，对图片进行缩放
	if resize_rate != 1:
		img = cv2.resize(img, (int(pic_width * resize_rate), int(pic_height * resize_rate)), interpolation=cv2.INTER_LANCZOS4)
		pic_height, pic_width = img.shape[:2]
	return img, pic_height, pic_width


# 车牌图像再处理
def car_pic_process(img, old_img, pic_height, pic_width):
	# 查找图像边缘整体形成的矩形区域
	try:
		contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	except ValueError:
		image, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Min_Area]

	# 排除不是车牌的矩形区域
	car_contours = []
	for cnt in contours:
		rect = cv2.minAreaRect(cnt)
		area_width, area_height = rect[1]
		if area_width < area_height:
			area_width, area_height = area_height, area_width
		wh_ratio = area_width / area_height

		# 2到5.5是车牌的长宽比，其余的矩形排除
		if (wh_ratio > 2) and (wh_ratio < 5.5):
			car_contours.append(rect)

	card_images = []
	for rect in car_contours:
		# 根据矩形区域的角度矫正倾斜
		if (rect[2] > -1) and (rect[2] < 1):
			angle = 1
		else:
			angle = rect[2]
		rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), angle)

		# 计算矩形区域的四个顶点
		box = cv2.boxPoints(rect)
		height_point = right_point = [0, 0]
		left_point = low_point = [pic_width, pic_height]
		for point in box:
			if left_point[0] > point[0]:
				left_point = point
			if low_point[1] > point[1]:
				low_point = point
			if height_point[1] < point[1]:
				height_point = point
			if right_point[0] < point[0]:
				right_point = point

		# 正角度矫正
		if left_point[1] <= right_point[1]:
			new_right_point = [right_point[0], height_point[1]]
			pts2 = np.float32([left_point, height_point, new_right_point])
			pts1 = np.float32([left_point, height_point, right_point])
			M = cv2.getAffineTransform(pts1, pts2)
			dst = cv2.warpAffine(old_img, M, (pic_width, pic_height))
			point_limit(new_right_point)
			point_limit(height_point)
			point_limit(left_point)
			card_img = dst[int(left_point[1]):int(height_point[1]), int(left_point[0]):int(new_right_point[0])]
			card_images.append(card_img)

		# 负角度矫正
		elif left_point[1] > right_point[1]:
			new_left_point = [left_point[0], height_point[1]]
			pts2 = np.float32([new_left_point, height_point, right_point])
			pts1 = np.float32([left_point, height_point, right_point])
			M = cv2.getAffineTransform(pts1, pts2)
			dst = cv2.warpAffine(old_img, M, (pic_width, pic_height))
			point_limit(right_point)
			point_limit(height_point)
			point_limit(new_left_point)
			card_img = dst[int(right_point[1]):int(height_point[1]), int(new_left_point[0]):int(right_point[0])]
			card_images.append(card_img)
	return card_images


# 获取颜色hsv范围
def get_limits(color):
	c = np.uint8([[color]])
	hsvc = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

	lower_limit = hsvc[0][0][0] - 10, 100, 100
	upper_limit = hsvc[0][0][0] + 10, 255, 255

	lower_limit = np.array(lower_limit, dtype=np.uint8)
	upper_limit = np.array(upper_limit, dtype=np.uint8)

	return lower_limit, upper_limit


# 读取一个模板地址与图片进行匹配，返回得分
def template_score(template,image):
	# 将模板进行格式转换
	template_img = cv2.imdecode(np.fromfile(template,dtype=np.uint8), 1)
	template_img = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
	# 模板图像阈值化处理——获得黑白图
	ret, template_img = cv2.threshold(template_img, 0, 255, cv2.THRESH_OTSU)
	image_ = image.copy()
	# 获得待检测图片的尺寸
	height, width = image_.shape
	# 将模板resize至与图像一样大小
	template_img = cv2.resize(template_img, (width, height))
	# 模板匹配，返回匹配得分
	result = cv2.matchTemplate(image_, template_img, cv2.TM_CCOEFF)
	return result[0][0]


# 获取字符模板
chinese_words_list = get_chinese_words_list()
eng_words_list = get_eng_words_list()
eng_num_words_list = get_eng_num_words_list()


# 车牌识别器类
class CardPredictor:
	def __init__(self):
		# 车牌识别的部分参数保存在js中，便于根据图片分辨率做调整
		f = open('config.js')
		j = json.load(f)
		for c in j["config"]:
			if c["open"]:
				self.cfg = c.copy()
				break
		else:
			raise RuntimeError('没有设置有效配置参数')

	# 精确定位车牌
	def accurate_place(self, card_img_hsv, limit1, limit2, color):
		row_num, col_num = card_img_hsv.shape[:2]
		xl = col_num
		xr = 0
		yh = 0
		yl = row_num
		row_num_limit = self.cfg["row_num_limit"]
		col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5  # 绿色渐变
		for i in range(row_num):
			count = 0
			for j in range(col_num):
				h = card_img_hsv.item(i, j, 0)
				s = card_img_hsv.item(i, j, 1)
				v = card_img_hsv.item(i, j, 2)
				if limit1 < h <= limit2 and 34 < s and 46 < v:
					count += 1
			if count > col_num_limit:
				if yl > i:
					yl = i
				if yh < i:
					yh = i
		for j in range(col_num):
			count = 0
			for i in range(row_num):
				h = card_img_hsv.item(i, j, 0)
				s = card_img_hsv.item(i, j, 1)
				v = card_img_hsv.item(i, j, 2)
				if limit1 < h <= limit2 and 34 < s and 46 < v:
					count += 1
			if count > row_num - row_num_limit:
				if xl > j:
					xl = j
				if xr < j:
					xr = j
		return xl, xr, yh, yl

	# 识别车牌号码
	def predict(self, car_pic, resize_rate=1):
		img, pic_height, pic_width = car_pic_preprocess(car_pic, resize_rate)
		blur = self.cfg["blur"]

		# 高斯去噪
		if blur > 0:
			img = cv2.GaussianBlur(img, (blur, blur), 0)  # 图片分辨率调整
		old_img = img
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		#  去掉图像中不会是车牌的区域
		kernel = np.ones((20, 20), np.uint8)
		img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
		img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0)

		# 找到图像边缘
		ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		img_edge = cv2.Canny(img_thresh, 100, 200)

		# 使用开运算和闭运算让图像边缘成为一个整体
		kernel = np.ones((self.cfg["morphologyr"], self.cfg["morphologyc"]), np.uint8)
		img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
		img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)

		# 车牌图像再处理
		card_images = car_pic_process(img_edge2, old_img, pic_height, pic_width)

		# 颜色定位
		colors = []
		for card_index, card_img in enumerate(card_images):
			green = yellow = blue = black = white = 0
			card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)

			# 处理转换失败的可能
			if card_img_hsv is None:
				continue
			row_num, col_num = card_img_hsv.shape[:2]
			card_img_count = row_num * col_num

			# 根据HSV颜色空间统计各颜色的像素数量
			for i in range(row_num):
				for j in range(col_num):
					h = card_img_hsv.item(i, j, 0)
					s = card_img_hsv.item(i, j, 1)
					v = card_img_hsv.item(i, j, 2)
					# 图片分辨率调整
					if (11 < h <= 34) and (s > 34):
						yellow += 1
					elif (35 < h <= 99) and (s > 34):
						green += 1
					elif (99 < h <= 124) and (s > 34):
						blue += 1
					elif (0 <= h < 180) and (0 <= s < 43) and (0 <= v < 46):
						black += 1
					elif (0 <= h < 180) and (0 <= s < 43) and (221 <= v < 225):
						white += 1

			# 根据颜色像素数量确定车牌颜色
			plate_color = "no"
			limit1 = limit2 = 0
			if yellow*2 >= card_img_count:
				plate_color = "yellow"
				limit1 = 11
				limit2 = 34  # 黄色车牌的HSV阈值范围
			elif green*2 >= card_img_count:
				plate_color = "green"
				limit1 = 35
				limit2 = 99  # 绿色车牌的HSV阈值范围
			elif blue*2 >= card_img_count:
				plate_color = "blue"
				limit1 = 100
				limit2 = 124  # 蓝色车牌的HSV阈值范围
			colors.append(plate_color)
			if limit1 == 0:
				continue

			# 根据车牌颜色进行再定位，缩小边缘非车牌边界
			xl, xr, yh, yl = self.accurate_place(card_img_hsv, limit1, limit2, plate_color)
			if yl == yh and xl == xr:
				continue
			need_accurate = False
			if yl >= yh:
				yl = 0
				yh = row_num
				need_accurate = True
			if xl >= xr:
				xl = 0
				xr = col_num
				need_accurate = True

			# 根据条件选择车牌区域
			adjust_green_plate = plate_color != "green" or yl < (yh - yl) // 4
			if adjust_green_plate:
				card_images[card_index] = card_img[yl:yh, xl:xr]
			else:
				card_images[card_index] = card_img[yl - (yh - yl) // 4:yh, xl:xr]

			# 如果需要，对车牌图像进行进一步的精确定位
			if need_accurate:
				card_img = card_images[card_index]
				card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
				xl, xr, yh, yl = self.accurate_place(card_img_hsv, limit1, limit2, plate_color)
				if yl == yh and xl == xr:
					continue
				if yl >= yh:
					yl = 0
					yh = row_num
				if xl >= xr:
					xl = 0
					xr = col_num

			# 根据条件选择车牌区域
			height = yh - yl
			need_adjustment = plate_color != "green" or yl < height // 4
			if need_adjustment:
				card_images[card_index] = card_img[yl:yh, xl:xr]
			else:
				card_images[card_index] = card_img[yl - height // 4:yh, xl:xr]

		# 字符识别
		print("字符识别中,请等待")
		predict_result = []
		roi = None
		card_color = None
		for i, color in enumerate(colors):
			if color in ("blue", "yellow", "green"):
				card_img = card_images[i]
				gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
				# 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
				if color == "green" or color == "yellow":
					gray_img = cv2.bitwise_not(gray_img)
				ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

				# 查找水平直方图波峰，确定车牌区域
				x_histogram = np.sum(gray_img, axis=1)
				x_min = np.min(x_histogram)
				x_average = np.sum(x_histogram)/x_histogram.shape[0]
				x_threshold = (x_min + x_average)/2
				wave_peaks = find_waves(x_threshold, x_histogram)

				# 波峰数量不足，跳过当前车牌图像
				if len(wave_peaks) == 0:
					continue

				# 确定车牌区域
				wave = max(wave_peaks, key=lambda x: x[1]-x[0])
				gray_img = gray_img[wave[0]:wave[1]]

				# 查找垂直直方图波峰，确定车牌中的字符
				row_num, col_num = gray_img.shape[:2]
				gray_img = gray_img[1:row_num-1]
				y_histogram = np.sum(gray_img, axis=0)
				y_min = np.min(y_histogram)
				y_average = np.sum(y_histogram)/y_histogram.shape[0]
				y_threshold = (y_min + y_average)/5  # U和0要求阈值偏小，否则U和0会被分成两半

				wave_peaks = find_waves(y_threshold, y_histogram)

				# 波峰数量不足，跳过当前车牌图像
				if len(wave_peaks) <= 6:
					continue

				# 确定车牌的字符区域
				wave = max(wave_peaks, key=lambda x: x[1]-x[0])
				max_wave_dis = wave[1] - wave[0]

				# 判断是否是左侧车牌边缘
				if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis/3 and wave_peaks[0][0] == 0:
					wave_peaks.pop(0)

				# 组合分离汉字
				cur_dis = 0
				for i, wave in enumerate(wave_peaks):
					if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
						break
					else:
						cur_dis += wave[1] - wave[0]
				if i > 0:
					wave = (wave_peaks[0][0], wave_peaks[i][1])
					wave_peaks = wave_peaks[i+1:]
					wave_peaks.insert(0, wave)

				# 去除车牌上的分隔点
				point = wave_peaks[2]
				if point[1] - point[0] < max_wave_dis/3:
					point_img = gray_img[:, point[0]:point[1]]
					if np.mean(point_img) < 255/5:
						wave_peaks.pop(2)

				# 波峰数量不足，跳过当前车牌图像
				if len(wave_peaks) <= 6:
					continue
				part_cards = seperate_card(gray_img, wave_peaks)
				num = 3
				for i, part_card in enumerate(part_cards):
					# 可能是固定车牌的铆钉
					if np.mean(part_card) < 255/5:
						print("a point")
						continue
					part_card_old = part_card
					w = part_card.shape[1] // 3
					part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
					part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)

					if i == 0:
						best_score = []
						for chinese_words in chinese_words_list:
							score = []
							for chinese_word in chinese_words:
								result = template_score(chinese_word, part_card)
								score.append(result)
							best_score.append(max(score))
						i = best_score.index(max(best_score))
						charactor = template[34 + i]
						predict_result.append(charactor)
						print(f"车牌第1个字符为：{charactor}")
						continue
					if i == 1:
						best_score = []
						for eng_word_list in eng_words_list:
							score = []
							for eng_word in eng_word_list:
								result = template_score(eng_word, part_card)
								score.append(result)
							best_score.append(max(score))
						i = best_score.index(max(best_score))
						charactor = template[10 + i]
						predict_result.append(charactor)
						print(f"车牌第2个字符为：{charactor}")
						continue
					else:
						best_score = []
						for eng_num_word_list in eng_num_words_list:
							score = []
							for eng_num_word in eng_num_word_list:
								result = template_score(eng_num_word, part_card)
								score.append(result)
							best_score.append(max(score))
						i = best_score.index(max(best_score))
						charactor = template[i]

						# 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
						if charactor == "1" and i == len(part_cards) - 1:
							if part_card_old.shape[0] / part_card_old.shape[1] >= 8:  # 1太细，认为是边缘
								print("an edge")
								continue
						else:
							predict_result.append(charactor)
							print(f"车牌第{num}个字符为：{charactor}")
							num += 1
				roi = card_img
				card_color = color
				break
		print(f"车牌颜色为：{card_color}")
		# 返回识别到的字符、定位的车牌图像、车牌颜色
		return predict_result, roi, card_color

	# 识别港澳车牌
	def predict_hk(self, car_pic, resize_rate=1):
		img, pic_height, pic_width = car_pic_preprocess(car_pic, resize_rate)
		blur = self.cfg["blur"]

		# 高斯去噪
		if blur > 0:
			img = cv2.GaussianBlur(img, (blur, blur), 0)  # 图片分辨率调整
		old_img = img

		# 获取港澳车牌黄色hsv范围
		yellow = [0, 255, 255]
		lower_limit, upper_limit = get_limits(color=yellow)

		hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv_img, lower_limit, upper_limit)

		ret, img = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)

		# 车牌再处理
		card_images = car_pic_process(img, old_img, pic_height, pic_width)

		# 黄色车牌识别
		predict_result = []
		roi = None
		card_color = "hk_yellow"
		for card_img in card_images:
			gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
			if card_color == "hk_yellow":
				gray_img = cv2.bitwise_not(gray_img)
			ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

			# 查找水平直方图波峰，确定车牌区域
			x_histogram = np.sum(gray_img, axis=1)
			x_min = np.min(x_histogram)
			x_average = np.sum(x_histogram) / x_histogram.shape[0]
			x_threshold = (x_min + x_average) / 2
			wave_peaks = find_waves(x_threshold, x_histogram)

			# 波峰数量不足，跳过当前车牌图像
			if len(wave_peaks) == 0:
				continue

			# 确定车牌区域
			wave = max(wave_peaks, key=lambda x: x[1] - x[0])
			gray_img = gray_img[wave[0]:wave[1]]

			# 查找垂直直方图波峰，确定车牌中的字符
			row_num, col_num = gray_img.shape[:2]
			gray_img = gray_img[1:row_num - 1]
			y_histogram = np.sum(gray_img, axis=0)
			y_min = np.min(y_histogram)
			y_average = np.sum(y_histogram) / y_histogram.shape[0]
			y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半

			wave_peaks = find_waves(y_threshold, y_histogram)

			# 波峰数量不足，跳过当前车牌图像
			if len(wave_peaks) < 3:
				continue

			# 确定车牌的字符区域
			wave = max(wave_peaks, key=lambda x: x[1] - x[0])
			max_wave_dis = wave[1] - wave[0]

			# 判断是否是左侧车牌边缘
			if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
				wave_peaks.pop(0)

			# 组合分离汉字
			cur_dis = 0
			for i, wave in enumerate(wave_peaks):
				if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
					break

			# 去除车牌上的分隔点
			point = wave_peaks[2]
			if point[1] - point[0] < max_wave_dis / 3:
				point_img = gray_img[:, point[0]:point[1]]
				if np.mean(point_img) < 255 / 5:
					wave_peaks.pop(2)

			# 波峰数量不足，跳过当前车牌图像
			if len(wave_peaks) < 3:
				continue
			part_cards = seperate_card(gray_img, wave_peaks)
			num = 1
			for i, part_card in enumerate(part_cards):
				# 可能是固定车牌的铆钉
				if np.mean(part_card) < 255 / 5:
					print("a point")
					continue
				part_card_old = part_card
				w = part_card.shape[1] // 3
				part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
				part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)

				# 字符匹配
				best_score = []
				for eng_num_word_list in eng_num_words_list:
					score = []
					for eng_num_word in eng_num_word_list:
						result = template_score(eng_num_word, part_card)
						score.append(result)
					best_score.append(max(score))
				i = best_score.index(max(best_score))
				charactor = template[i]

				# 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
				if charactor == "1" and i == len(part_cards) - 1:
					if part_card_old.shape[0] / part_card_old.shape[1] >= 8:  # 1太细，认为是边缘
						print("an edge")
						continue
				else:
					predict_result.append(charactor)
					print(f"车牌第{num}个字符为：{charactor}")
					num += 1
			roi = card_img
			break
		print(f"车牌颜色为：{card_color}")
		# 返回识别到的字符、定位的车牌图像、车牌颜色
		return predict_result, roi, card_color
