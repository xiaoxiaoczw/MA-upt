
from cv2 import VideoCapture
from cv2 import imwrite


def save_image(image, addr, num):
    address = addr + str(num) + '.jpg'
    imwrite(address, image)


if __name__ == '__main__':

    video_path = "./video/cooking.mp4"  # video path
    out_path = "./frames/img_"  # saved frame path

    # is_all_frame = False
    # sta_frame = 1
    # end_frame = 40

    is_all_frame = True

    ######
    time_interval = 10  # time_interval

    # read video
    videoCapture = VideoCapture(video_path)

    # write frame
    success, frame = videoCapture.read()
    print(success)

    i = 0
    j = 0
    if is_all_frame:
        time_interval = 10

    while success:
        i = i + 1
        if (i % time_interval == 0):
            if is_all_frame == False:
                if i >= sta_frame and i <= end_frame:
                    j = j + 1
                    print('save frame:', i)
                    save_image(frame, out_path, j)
                elif i > end_frame:
                    break
            else:
                j = j + 1
                print('save frame:', i)
                save_image(frame, out_path, j)

        success, frame = videoCapture.read()