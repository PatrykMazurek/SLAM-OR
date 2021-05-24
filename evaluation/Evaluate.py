import cv2
import numpy as np

recording_id = "09"

arr_recording_id = ["09","22","23","35","39","46","61","64","84","91"]
arr_dbscan_eps = [0.5,0.75,1,1.25,1.5]
arr_dbscan_min = [3, 5, 7]

a = 0
b = 0
c = 0

for bb in range(len(arr_dbscan_eps)):
    for cc in range(len(arr_dbscan_min)):
        for aa in range(len(arr_recording_id)):

            recording_id = arr_recording_id[aa]
            dbscan_eps = arr_dbscan_eps[bb]
            dbscan_min = arr_dbscan_min[cc]

            fig_ref = "results/reference_id=" + recording_id + ".png"
            fig_res = "results/results_id=" + recording_id + ",eps=" + str(dbscan_eps) + ",min=" + str(dbscan_min) + ".png"
            print(fig_ref + " " + fig_res)
            res = cv2.imread(fig_res)
            ref = cv2.imread(fig_ref)

            le = np.logical_and(res, ref)
            le = np.zeros(res.shape)

            P = 0
            TP = 0
            FP = 0
            FN = 0

            for a  in range(le.shape[0]):
                for b in range(le.shape[1]):

                    if ref[a,b, 0] > 0:
                        P = P + 1
                    if res[a,b, 0] > 0 and ref[a,b,0] > 0:
                        le[a, b] = 255
                        TP = TP + 1
                    if res[a,b, 0] > 0 and ref[a,b, 0] == 0:
                        FP = FP + 1
                    if res[a,b, 0] == 0 and ref[a,b, 0] > 0:
                        FN = FN + 1

            #https://en.wikipedia.org/wiki/Confusion_matrix
            import math
            #print(le)
            TPR = TP / P
            PPV = TP / (TP + FP)
            FNR = FN / (FN + TP)
            F1 = 2 * TP / ((2 * TP) + FP + FN)
            FM = math.sqrt(PPV * TPR)

            my_str = recording_id + "\t" + str(dbscan_eps) + "\t" + str(dbscan_min) + "\t"  + str(TPR) + "\t" + str(PPV) + "\t" + str(FNR) + "\t" + str(F1) + "\t" + str(FM)
            print(my_str)

            with open('results/results.txt', 'a') as the_file:
                the_file.write(my_str + '\n')
            """
            print("TPR=" + str(TP / P))
            print("PPV=" + str(TP / (TP + FP)))
            print("FNR=" + str(FN / (FN + TP)))
            print("F1-score=" + str(2 * TP / (2 * TP + FP +FN)))
            print("FP=" + str(math.sqrt(PPV * TPR)))
            """
            #cv2.imshow("le",le)
            #cv2.waitKey()