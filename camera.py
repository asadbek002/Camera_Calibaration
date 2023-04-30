import numpy as np
import cv2 as cv

# 체스보드 패턴
board_size = (10, 6)
square_size = 2.5  # cm

# 체스보드 모서리 좌표 생성
objp = np.zeros((np.prod(board_size), 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
objp *= square_size

# 체스보드 모서리를 찾기 위한 termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 저장된 이미지 파일 목록
image_files = ['clib1.jpg', 'clib 2.jpg', 'clib3.jpg', 'clib4.jpg']

# 체스보드 모서리 좌표와 이미지에서 검출된 좌표 저장용 리스트
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane.

# 이미지 파일을 하나씩 읽어들이면서
for fname in image_files:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 체스보드의 모서리를 찾음
    ret, corners = cv.findChessboardCorners(gray, board_size, None)

    # 모서리가 검출되면 objpoints와 imgpoints에 각각 추가함
    if ret == True:
        objpoints.append(objp)

        # 모서리 좌표를 더 정확히 찾음
        corners2 = cv.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # 검출 결과를 이미지에 표시
        img = cv.drawChessboardCorners(img, board_size, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

# 카메라 캘리브레이션
ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

# 캘리브레이션 결과 출력
print('ret:', ret)
print('K:', K)
print('dist:', dist)
print('rvecs:', rvecs)
print('tvecs:', tvecs)

# 포즈 추정
img = cv.imread('clib1.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, corners = cv.findChessboardCorners(gray, board_size, None)
if ret == True:
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # 포즈 추정
    ret, rvecs, tvecs = cv.solvePnP(objp, corners2, K, dist)

    # 이미지에 좌표축 그리기
    axis = np.float32([[5, 0, 0], [0, 5, 0], [0, 0, -5]])