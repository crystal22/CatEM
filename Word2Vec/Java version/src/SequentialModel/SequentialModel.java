package SequentialModel;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import HierarchicalModel.Skipgram;

public class SequentialModel {

    public static void main(String[] args) throws IOException {
        SequentialModel sm = new SequentialModel();
        double learnRate = 0.025;

        //contextWindowSize=5, perform the best
        for (int contextWindowSize = 1; contextWindowSize <= 6; contextWindowSize++) {
            String country = "MY";
            String trainPath = "C:\\jianguoyun\\JianguoShare\\2学术论文\\2019HierarchicalCategoryEmbedding\\data\\embeddingData\\CheckinCategoryIDTimeSequence" + country + "50.csv";
            ArrayList<String> trainList = new ArrayList<String>();
            sm.getTrainList(trainPath, trainList, contextWindowSize);
//			String inputPath="C:\\Users\\chenm\\Desktop\\input\\sgJPinput.csv";
//			sm.list2File(inputPath, trainList);
            System.out.println("train file complete");


            Skipgram sk = new Skipgram();
            int negativeNum = 1;
            int[] vecSize = {50};
            for (int i = 0; i < vecSize.length; i++) {
                String contextPath = "C:\\jianguoyun\\JianguoShare\\2学术论文\\2019HierarchicalCategoryEmbedding\\data\\embeddings\\" + country + "\\SequentialModel\\contextEmbedding" + contextWindowSize + "#" + vecSize[i] + ".csv";
                String targetPath = "C:\\jianguoyun\\JianguoShare\\2学术论文\\2019HierarchicalCategoryEmbedding\\data\\embeddings\\" + country + "\\SequentialModel\\targetEmbedding" + contextWindowSize + "#" + vecSize[i] + ".csv";
                sk.skipgram(trainList, negativeNum, vecSize[i], contextPath, targetPath, learnRate);
            }

        }
    }


    private void getTrainList(String trainPath, ArrayList<String> trainList, int contextWindowSize) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(trainPath));
        while (true) {
            String line = br.readLine();
            if (line == null)
                break;
            String[] temp = line.split(",");
            for (int i = 1; i < temp.length - 1; i++) {
                String targetCategory = temp[i].split("#")[0];
                // context category
                ArrayList<String> contextCategortList = new ArrayList<String>();
                for (int j = i - 1; j >= Math.max(0, i - contextWindowSize); j--) {
                    contextCategortList.add(temp[j].split("#")[0]);
                }
                for (int j = i + 1; j <= Math.min(temp.length - 1, i + contextWindowSize); j++) {
                    contextCategortList.add(temp[j].split("#")[0]);
                }

                String record = targetCategory + ",";

                for (int j = 0; j < contextCategortList.size() - 1; j++) {
                    record += contextCategortList.get(j) + "#";
                }
                record += contextCategortList.get(contextCategortList.size() - 1);
                trainList.add(record);
            }
        }
        br.close();
    }
}
