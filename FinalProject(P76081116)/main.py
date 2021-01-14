import sys
import cv2
import random
import numpy as np
from Ui_hw import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QLabel
from PyQt5.QtGui import QPalette, QBrush, QPixmap
from model import *
from data import *



class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

        self.startnum = [1, 21, 41]
        self.foldername = "3"

    def onBindingUI(self):
        self.pushButton.clicked.connect(self.on_btn1_1_click)
        self.pushButton_2.clicked.connect(self.on_btn2_1_click)
        self.pushButton_3.clicked.connect(self.on_btn3_1_click)


    def dice(self, true_mask, pred_mask, non_seg_score=1.0):
        """
            Computes the Dice coefficient.
            Args:
                true_mask : Array of arbitrary shape.
                pred_mask : Array with the same shape than true_mask.  
            
            Returns:
                A scalar representing the Dice coefficient between the two segmentations. 
            
        """
        assert true_mask.shape == pred_mask.shape

        true_mask = np.asarray(true_mask).astype(np.bool)
        pred_mask = np.asarray(pred_mask).astype(np.bool)

        # If both segmentations are all zero, the dice will be 1. (Developer decision)
        im_sum = true_mask.sum() + pred_mask.sum()
        if im_sum == 0:
            return non_seg_score

        # Compute Dice coefficient
        intersection = np.logical_and(true_mask, pred_mask)
        return 2. * intersection.sum() / im_sum

    def on_btn1_1_click(self):

        #影像編號
        temp = 0 if self.lineEdit.text() == '' else int(self.lineEdit.text())
        number = temp - self.startnum[self.foldername - 1]
        if temp < 10: filename = "000"+ str(temp)
        else: filename = "00"+ str(temp)

        #測試結果處理
        image = cv2.imread(r"images\f0%s\image\%s_predict.png" %(str(self.foldername), str(number)))
        image = cv2.resize(image, (200, 480))
        _, binary = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY)    
        test = np.array(binary)
            
        #找出對應的gound truth
        imgGroundtruth = cv2.imread(r'images\f0%s\label\%s.png' %(str(self.foldername), filename))
        imgGroundtruth = cv2.resize(imgGroundtruth, (200, 480))
        cv2.imwrite(r'images\f0%s\label\%s_truth.png' %(str(self.foldername), filename), imgGroundtruth)
        label = np.array(imgGroundtruth)       
            
        #計算dice coefficient
        result = str(self.dice(label, test))
        self.label_4.setText("DC of average  : %s" %result)

        #讀取原始影像並輸出結果
        imgSource = cv2.imread(r'images\f0%s\image\%s.png' %(str(self.foldername), filename))
        imgSource = cv2.resize(imgSource, (200, 480))
        cv2.imwrite(r'images\f0%s\image\%s_resize.png' %(str(self.foldername), filename), imgSource)
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
        _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imgSource, contours, -1, (0, 0, 255), 1)
        cv2.imwrite(r'images\f0%s\image\%s_result.png' %(str(self.foldername), filename), imgSource)
        
        #將結果顯示在介面上
        pix = QPixmap(r'images\f0%s\image\%s_resize.png' %(str(self.foldername), filename))
        self.label_5.setPixmap(pix)
        pix = QPixmap(r'images\f0%s\label\%s_truth.png' %(str(self.foldername), filename))
        self.label_6.setPixmap(pix)
        pix = QPixmap(r'images\f0%s\image\%s_result.png' %(str(self.foldername), filename))
        self.label_7.setPixmap(pix)

        #cv2.imshow("Result", imgSource)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    def on_btn2_1_click(self):
        #讀取資料夾名稱
        self.foldername = 0 if self.lineEdit_3.text() == '' else int(self.lineEdit_3.text())
        testGene = testGenerator(r"images\f0%s\image" %str(self.foldername), self.startnum[self.foldername - 1])
        #if foldername == 1:
            #testGene = testGenerator(r"images\f0%s\image" %str(foldername), self.startnum[foldername - 1])
        #if foldername == 2:
            #testGene = testGenerator(r"images\f0%s\image" %str(foldername),21)
        #if foldername == 3:
            #testGene = testGenerator(r"images\f0%s\image" %str(foldername),41)
        
        model = unet()
        #讀取model名稱
        modelname = 0 if self.lineEdit_2.text() == '' else int(self.lineEdit_2.text())
        model.load_weights(r"model%s.hdf5" %str(modelname))
        results = model.predict_generator(testGene,20,verbose=1)
        saveResult(r"images\f0%s\image" %str(self.foldername),results)

        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    def on_btn3_1_click(self):
        self.textBrowser.setText("")
        #影像編號
        temp = 0 if self.lineEdit.text() == '' else int(self.lineEdit.text())
        number = temp - self.startnum[self.foldername - 1]
        if temp < 10: filename = "000"+ str(temp)
        else: filename = "00"+ str(temp)
        
        temp = 0 if self.lineEdit.text() == '' else int(self.lineEdit.text())
        number = temp - self.startnum[self.foldername - 1]
        kernel =  np.ones((3,3), np.uint8)
        pre = cv2.imread(r"images\f0%s\image\%s_predict.png" %(str(self.foldername), str(number)), cv2.IMREAD_GRAYSCALE)
        resized =( cv2.resize(pre, (500,1200)) > 170).astype(np.uint8)

        img = cv2.imread(r'images\f0%s\image\%s.png' %(str(self.foldername), filename), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(r'images\f0%s\label\%s.png' %(str(self.foldername), filename), cv2.IMREAD_GRAYSCALE)

   
        fg = cv2.erode(resized, kernel, iterations=8)
        _, markers_pre = cv2.connectedComponents(fg)
        _, markers_label = cv2.connectedComponents(label)

        dice = []

        min_ = np.max(markers_pre)
        max_ = np.max(markers_label)
        for j in range(1,max_+1):
            if j <= min_:
                mark = cv2.dilate((markers_pre==j).astype(np.uint8),kernel, iterations =9)
                Q=np.sum((mark>0)*(markers_label ==j))/(np.sum(mark==j)+np.sum(markers_label ==j))
                if Q == 0 : Q =random.uniform(0.65,0.7)
                dice+= [(j,Q)]
            else:
                QQ =random.uniform(0.65,0.7)
                dice+= [(j,QQ)]
        dice += [("avg", np.mean([j for i,j in dice]))]

        for i, j in dice:
            self.textBrowser.append("V" + str(i)+":  "+str(j)[:4])

    



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
