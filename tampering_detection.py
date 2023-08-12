import numpy as np
import cv2
import os
from absl import flags,app,logging
from skimage.metrics import structural_similarity as ssim



class tampering_detection :
    
    def __init__(self,blur_thresh,occ_thresh,area_thresh,ssim_thresh):
        self.occ_flag=0
        self.scene_change_flag =0
        self.movement_flag=0
        self.defocusing_flag=0
        self.fgbg = cv2.createBackgroundSubtractorMOG2(1000)
        self.focus_mean_20 = 0
        self.focus_mean_100 = 0
        self.occ_mean_20 = 0
        self.occ_mean_100 = 0
        self.kernel = np.ones(shape=(5,5))
        self.blur_thresh = blur_thresh
        self.occ_thresh = occ_thresh
        self.area_thresh = area_thresh
        self.ssim_thresh = ssim_thresh
        self.count = 0
        if not os.path.exists('tampered_frames'):
            os.makedirs('tampered_frames')

    def movement_detection(self):
        a = 0
        self.movement_flag = 0
        bounding_rect = []
        self.fgmask = self.fgbg.apply(self.frame)
        self.bgmask = self.fgbg.getBackgroundImage()
        if FLAGS.display_fg_bg :
            cv2.imshow('bg',self.bgmask)
            cv2.imshow('fg', self.fgmask)
        self.fgmask = cv2.erode(self.fgmask, self.kernel, iterations=5)
        self.fgmask = cv2.dilate(self.fgmask, self.kernel, iterations=5)
        contours, _ = cv2.findContours(self.fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        

        for i in range(0, len(contours)):
            cntr = cv2.boundingRect(contours[i])
            bounding_rect.append(cntr)
            x, y, w, h = cntr
            cv2.rectangle(self.fgmask, (x, y), (x + w, y + h), (36, 255, 12), 2)

        for i in range(0, len(contours)):
            if bounding_rect[i][2] >= 40 or bounding_rect[i][3] >= 40:
                a = a + (bounding_rect[i][2]) * bounding_rect[i][3]
            if (a >= int(self.frame.shape[0]) * int(self.frame.shape[1]) * self.area_thresh):
                self.movement_flag = 1



    def defocusing_detection(self):

        self.defocusing_flag = 0
        var = cv2.Laplacian(self.frame, cv2.CV_64F).var()
        self.focus_mean_100 = self.focus_mean_100 * 0.98 + var * 0.02
        self.focus_mean_20 = self.focus_mean_20 * 0.9 + var * 0.1
        self.blur_score = (self.focus_mean_100 - self.focus_mean_20) / self.focus_mean_100


        if self.blur_score >= self.blur_thresh:
            self.defocusing_flag = 1

    def occlusion_detection(self):

        self.occ_flag = 0
        var = np.var(self.frame)
        self.occ_mean_100 = self.occ_mean_100 * 0.98 + var * 0.02
        self.occ_mean_20 = self.occ_mean_20 * 0.9 + var * 0.1
        self.occ = (self.occ_mean_100 - self.occ_mean_20) / self.occ_mean_100
        if self.occ >= self.occ_thresh:
            self.occ_flag = 1

    def scene_change_detection(self):

        self.scene_change_flag = 0
        self.ssim = ssim(self.frame,self.bgmask,multichannel=True)
        if self.ssim <= self.ssim_thresh :
            self.scene_change_flag = 1

    def detect(self,vid):

        if vid == "0":
            vid = int(vid)
        ret = True
        cap = cv2.VideoCapture(vid)
        if FLAGS.save_output :
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(FLAGS.output_path, fourcc, 20.0, (640, 480))

        while (ret):

            ret, self.frame = cap.read()
            if (self.frame is None):
                print("End of frame")
                break
            else:
                self.count += 1
                self.defocusing_detection()
                self.movement_detection()
                self.occlusion_detection()
                if self.count % 100 == 0 :
                  self.scene_change_detection()
                  self.count = 0


                if self.movement_flag == 1 or self.defocusing_flag == 1 or self.occ_flag == 1 or self.scene_change_flag == 1:

                    cv2.putText(self.frame, "TAMPERING DETECTED", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)

                    if FLAGS.save_output:
                        self.count += 1
                        out.write(self.frame)

                cv2.imshow('frame', self.frame)


            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            if k == ord('r'):
                self.movement_flag = 0
                self.defocusing_flag = 0
                self.occ_flag =0
                self.scene_change_flag = 0

        cap.release()
        cv2.destroyAllWindows()



def main(_argv):
      logging.info("xxx-----  Setting Up Detector -----xxx")
      detector =tampering_detection(blur_thresh=FLAGS.blur_thresh,occ_thresh=FLAGS.occ_thresh,area_thresh = FLAGS.area_thresh,ssim_thresh = FLAGS.ssim_thresh)
      logging.info("occlusion threshold : {}".format(FLAGS.occ_thresh))
      logging.info("blur threshold : {}".format(FLAGS.blur_thresh))
      logging.info("area threshold : {}".format(FLAGS.area_thresh))
      logging.info("structural similarity threshold : {}".format(FLAGS.ssim_thresh))
      detector.detect(FLAGS.video_path)


if __name__ == '__main__':

    FLAGS = flags.FLAGS
    flags.DEFINE_boolean("save_output", True, help="If True save videos to specified directory")
    flags.DEFINE_float("occ_thresh", 0.5, help="threshold for occlusion detection")
    flags.DEFINE_float("blur_thresh", 0.5, help="threshhold for blur detection")
    flags.DEFINE_float("area_thresh", 0.33, help="threshhold for movement detection")
    flags.DEFINE_float("ssim_thresh", 0.6, help="threshold for image similarity")
    flags.DEFINE_string("video_path", "0", help="Path to video file , enter 0 for string")
    flags.DEFINE_string("output_path", "output.avi", help="Path to video file")
    flags.DEFINE_boolean("display_fg_bg", False, help="Path to video file")
    app.run(main)


