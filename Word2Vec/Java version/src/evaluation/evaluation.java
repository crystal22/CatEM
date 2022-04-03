package evaluation;

import sun.awt.windows.WPrinterJob;

import javax.swing.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

//import HierarchicalModel.SortByProbability;

/**
 * @author chen meng
 * @version：2019年11月30日 上午9:42:24
 */
public class evaluation {

    public static void main(String[] args) throws IOException {

        evaluation tt = new evaluation();

        String categoryPath = "E:\\IdeaProjects\\SequenceModel\\data\\category.csv";
        HashMap<String, String> categoryIDMap = new HashMap<String, String>();
        tt.getCategoryIDMap(categoryPath, categoryIDMap);

        System.out.println("wp \t lc \t jc_Sanchez \t jc_Seco \t lin_Sanchez \t lin_Seco");

        int vecSize[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
        String contextEmbeddingBase = "E:\\PycharmProjects\\TSMC\\vectors\\NYC\\temp\\";
        for (int i = 4; i <= 4; i++) {
            String embeddingPath = contextEmbeddingBase + "contextEmbedding5#" + vecSize[i] + "(1).csv";

//		String embeddingPath = "C:\\Users\\Administrator\\Desktop\\CategoryEmbedding\\embedding\\HCE_1\\JP\\contextEmbeddingJP40#1.csv";
            String categoryGroundtruthPath1 = "E:\\IdeaProjects\\SequenceModel\\data\\categorySIMwp.csv";
            String categoryGroundtruthPath2 = "E:\\IdeaProjects\\SequenceModel\\data\\categorySIMlc.csv";
            String categoryGroundtruthPath3 = "E:\\IdeaProjects\\SequenceModel\\data\\categorySIMjc_Sanchez.csv";
            String categoryGroundtruthPath4 = "E:\\IdeaProjects\\SequenceModel\\data\\categorySIMjc_Seco.csv";
            String categoryGroundtruthPath5 = "E:\\IdeaProjects\\SequenceModel\\data\\categorySIMlin_Sanchez.csv";
            String categoryGroundtruthPath6 = "E:\\IdeaProjects\\SequenceModel\\data\\categorySIMlin_Seco.csv";

            int[] topK = {10};
//			tt.computeAccuracy(embeddingPath, categoryGroundtruthPath1,topK);
//			tt.computeAccuracy(embeddingPath, categoryGroundtruthPath2,topK);
//			tt.computeAccuracy(embeddingPath, categoryGroundtruthPath3,topK);
//			tt.computeAccuracy(embeddingPath, categoryGroundtruthPath4,topK);
//			tt.computeAccuracy(embeddingPath, categoryGroundtruthPath5,topK);
//			tt.computeAccuracy(embeddingPath, categoryGroundtruthPath6,topK);

            tt.computeMRR(embeddingPath, categoryGroundtruthPath1, categoryIDMap);
            tt.computeMRR(embeddingPath, categoryGroundtruthPath2, categoryIDMap);
            tt.computeMRR(embeddingPath, categoryGroundtruthPath3, categoryIDMap);
            tt.computeMRR(embeddingPath, categoryGroundtruthPath4, categoryIDMap);
            tt.computeMRR(embeddingPath, categoryGroundtruthPath5, categoryIDMap);
            tt.computeMRR(embeddingPath, categoryGroundtruthPath6, categoryIDMap);

//			tt.computeMRRWordvec(embeddingPath, categoryGroundtruthPath1);
//			tt.computeMRRWordvec(embeddingPath, categoryGroundtruthPath2);
//			tt.computeMRRWordvec(embeddingPath, categoryGroundtruthPath3);
//			tt.computeMRRWordvec(embeddingPath, categoryGroundtruthPath4);
//			tt.computeMRRWordvec(embeddingPath, categoryGroundtruthPath5);
//			tt.computeMRRWordvec(embeddingPath, categoryGroundtruthPath6);
//			System.out.println();

        }
    }

    private void getCategoryIDMap(String path, HashMap<String, String> categoryIDMap) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(path));
        while (true) {
            String line = br.readLine();
            if (line == null) {
                break;
            }
            String[] temp = line.split(",");
            String category = temp[1];
            String ID = temp[0];
            categoryIDMap.put(category, ID);
        }
        br.close();
    }

    public void computeMRR(String embeddingPath, String categoryGroundtruthPath, HashMap<String, String> categoryNameIDMap) throws IOException {
        double MRR = 0;
        HashMap<String, double[]> categoryEmbeddingMap = new HashMap<String, double[]>();

        BufferedReader br = new BufferedReader(new FileReader(embeddingPath));
        while (true) {
            String line = br.readLine();
            if (line == null) {
                break;
            }
            String[] temp = line.split(",");
            String category = temp[0];
            double[] embedding = new double[temp.length - 1];
            for (int i = 1; i < temp.length; i++) {
                embedding[i - 1] = Double.parseDouble(temp[i]);
            }
            if (categoryNameIDMap.containsKey(category)) {
                categoryEmbeddingMap.put(categoryNameIDMap.get(category), embedding);
            }

        }
        br.close();

        HashMap<String, ArrayList<String>> categoryTopCategoryMap = new HashMap<String, ArrayList<String>>();
        br = new BufferedReader(new FileReader(categoryGroundtruthPath));
        while (true) {
            String line = br.readLine();
            if (line == null) {
                break;
            }
            String[] temp = line.split(",");
            String category = temp[0];
            if (categoryEmbeddingMap.containsKey(category)) {
                ArrayList<String> list = new ArrayList<String>();
                for (int i = 1; i < temp.length; i++) {
                    String topCategory = temp[i].split("@")[0];
                    if (categoryEmbeddingMap.containsKey(topCategory)) {
                        list.add(topCategory);
                    }
                }
                categoryTopCategoryMap.put(category, list);
            }
        }
        br.close();

        Set<String> categorySet = categoryEmbeddingMap.keySet();

        Iterator<String> itr = categorySet.iterator();
        while (itr.hasNext()) {
            String queryCategory = itr.next();
//			System.out.println(queryCategory);
            String topCategoryGroundtruth = categoryTopCategoryMap.get(queryCategory).get(0);
            double score = queryTop1(queryCategory, categoryEmbeddingMap, topCategoryGroundtruth);
//			if(score>0.99) {
//				System.out.println(queryCategory);
//			}
            MRR += score;
        }
        MRR = MRR / categorySet.size();
        System.out.print(MRR + ",");
//		pw.print(MRR + ",");
    }

    public void computeMRR(String embeddingPath, String categoryGroundtruthPath, HashMap<String, String> categoryNameIDMap, PrintWriter pw) throws IOException {
        double MRR = 0;
        HashMap<String, double[]> categoryEmbeddingMap = new HashMap<String, double[]>();

        BufferedReader br = new BufferedReader(new FileReader(embeddingPath));
        while (true) {
            String line = br.readLine();
            if (line == null) {
                break;
            }
            String[] temp = line.split(",");
            String category = temp[0];
            double[] embedding = new double[temp.length - 1];
            for (int i = 1; i < temp.length; i++) {
                embedding[i - 1] = Double.parseDouble(temp[i]);
            }
            if (categoryNameIDMap.containsKey(category)) {
                categoryEmbeddingMap.put(categoryNameIDMap.get(category), embedding);
            }

        }
        br.close();

        HashMap<String, ArrayList<String>> categoryTopCategoryMap = new HashMap<String, ArrayList<String>>();
        br = new BufferedReader(new FileReader(categoryGroundtruthPath));
        while (true) {
            String line = br.readLine();
            if (line == null) {
                break;
            }
            String[] temp = line.split(",");
            String category = temp[0];
            if (categoryEmbeddingMap.containsKey(category)) {
                ArrayList<String> list = new ArrayList<String>();
                for (int i = 1; i < temp.length; i++) {
                    String topCategory = temp[i].split("@")[0];
                    if (categoryEmbeddingMap.containsKey(topCategory)) {
                        list.add(topCategory);
                    }
                }
                categoryTopCategoryMap.put(category, list);
            }
        }
        br.close();

        Set<String> categorySet = categoryEmbeddingMap.keySet();

        Iterator<String> itr = categorySet.iterator();
        while (itr.hasNext()) {
            String queryCategory = itr.next();
//			System.out.println(queryCategory);
            String topCategoryGroundtruth = categoryTopCategoryMap.get(queryCategory).get(0);
            double score = queryTop1(queryCategory, categoryEmbeddingMap, topCategoryGroundtruth);
//			if(score>0.99) {
//				System.out.println(queryCategory);
//			}
            MRR += score;
        }
        MRR = MRR / categorySet.size();
        System.out.print(MRR + ",");
        pw.print(MRR + ",");
    }

    private void computeMRRWordvec(String embeddingPath, String categoryGroundtruthPath) throws IOException {
        double MRR = 0;
        HashMap<String, double[]> categoryEmbeddingMap = new HashMap<String, double[]>();

        BufferedReader br = new BufferedReader(new FileReader(embeddingPath));
        br.readLine();
        while (true) {
            String line = br.readLine();
            if (line == null) {
                break;
            }
            String[] temp = line.split(" ");
            String category = temp[0];
            double[] embedding = new double[temp.length - 1];
            for (int i = 1; i < temp.length; i++) {
                embedding[i - 1] = Double.parseDouble(temp[i]);
            }
            categoryEmbeddingMap.put(category, embedding);
        }
        br.close();

        HashMap<String, ArrayList<String>> categoryTopCategoryMap = new HashMap<String, ArrayList<String>>();
        br = new BufferedReader(new FileReader(categoryGroundtruthPath));
        while (true) {
            String line = br.readLine();
            if (line == null) {
                break;
            }
            String[] temp = line.split(",");
            String category = temp[0];
            if (categoryEmbeddingMap.containsKey(category)) {
                ArrayList<String> list = new ArrayList<String>();
                for (int i = 1; i < temp.length; i++) {
                    String topCategory = temp[i].split("@")[0];
                    if (categoryEmbeddingMap.containsKey(topCategory)) {
                        list.add(topCategory);
                    }
                }
                categoryTopCategoryMap.put(category, list);
            }
        }
        br.close();

        Set<String> categorySet = categoryEmbeddingMap.keySet();

        Iterator<String> itr = categorySet.iterator();
        while (itr.hasNext()) {
            String queryCategory = itr.next();
            String topCategoryGroundtruth = categoryTopCategoryMap.get(queryCategory).get(0);
            double score = queryTop1(queryCategory, categoryEmbeddingMap, topCategoryGroundtruth);
            MRR += score;
        }
        MRR = MRR / categorySet.size();
        System.out.print(MRR + "\t");
    }

    private void computeRandomMRR(String embeddingPath, String categoryGroundtruthPath) throws IOException {
        double MRR = 0;
        ArrayList<String> categoryList = new ArrayList<String>();
        HashSet<String> categorySet = new HashSet<String>();
        BufferedReader br = new BufferedReader(new FileReader(embeddingPath));
        while (true) {
            String line = br.readLine();
            if (line == null) {
                break;
            }
            String[] temp = line.split(",");
            String category = temp[0];
            categoryList.add(category);
            categorySet.add(category);
        }
        br.close();

        HashMap<String, ArrayList<String>> categoryTopCategoryMap = new HashMap<String, ArrayList<String>>();
        br = new BufferedReader(new FileReader(categoryGroundtruthPath));
        while (true) {
            String line = br.readLine();
            if (line == null) {
                break;
            }
            String[] temp = line.split(",");
            String category = temp[0];
            if (categorySet.contains(category)) {
                ArrayList<String> list = new ArrayList<String>();
                for (int i = 1; i < temp.length; i++) {
                    String topCategory = temp[i].split("@")[0];
                    if (categorySet.contains(topCategory)) {
                        list.add(topCategory);
                    }
                }
                categoryTopCategoryMap.put(category, list);
            }
        }
        br.close();

        Iterator<String> itr = categorySet.iterator();
        while (itr.hasNext()) {
            String queryCategory = itr.next();
            String topCategoryGroundtruth = categoryTopCategoryMap.get(queryCategory).get(0);
            double score = queryTop1Random(categoryList, topCategoryGroundtruth);
            MRR += score;
        }
        MRR = MRR / categorySet.size();
        System.out.print(MRR + "\t");
    }

    private void computeAccuracy(String embeddingPath, String categoryGroundtruthPath, int[] topK) throws IOException {

        HashMap<String, double[]> categoryEmbeddingMap = new HashMap<String, double[]>();

        BufferedReader br = new BufferedReader(new FileReader(embeddingPath));
        while (true) {
            String line = br.readLine();
            if (line == null) {
                break;
            }
            String[] temp = line.split(",");
            String category = temp[0];
            double[] embedding = new double[temp.length - 1];
            for (int i = 1; i < temp.length; i++) {
                embedding[i - 1] = Double.parseDouble(temp[i]);
            }
            categoryEmbeddingMap.put(category, embedding);
        }
        br.close();

        HashMap<String, ArrayList<String>> categoryTopCategoryMap = new HashMap<String, ArrayList<String>>();
        br = new BufferedReader(new FileReader(categoryGroundtruthPath));
        while (true) {
            String line = br.readLine();
            if (line == null) {
                break;
            }
            String[] temp = line.split(",");
            String category = temp[0];
            if (categoryEmbeddingMap.containsKey(category)) {
                ArrayList<String> list = new ArrayList<String>();
                for (int i = 1; i < temp.length; i++) {
                    String topCategory = temp[i].split("@")[0];
                    if (categoryEmbeddingMap.containsKey(topCategory)) {
                        list.add(topCategory);
                    }
                }
                categoryTopCategoryMap.put(category, list);
            }
        }
        br.close();

        double[] acc = new double[topK.length];
        Set<String> categorySet = categoryEmbeddingMap.keySet();
        Iterator<String> itr = categorySet.iterator();
        while (itr.hasNext()) {
            String queryCategory = itr.next();
            String topCategoryGroundtruth = categoryTopCategoryMap.get(queryCategory).get(0);
            double[] score = queryTopK(queryCategory, categoryEmbeddingMap, topCategoryGroundtruth, topK);
            for (int i = 0; i < score.length; i++) {
                acc[i] += score[i];
            }
        }

        for (int i = 0; i < acc.length; i++) {
            acc[i] = acc[i] / categorySet.size();
            System.out.print(acc[i] + "\t");
        }

    }

    private double[] queryTopK(String queryCategory, HashMap<String, double[]> categoryEmbeddingMap, String topCategory,
                               int[] topK) {
        double[] score = new double[topK.length];
        double[] queryEmbedding = categoryEmbeddingMap.get(queryCategory);
        ArrayList<String> rankList = new ArrayList<String>();
        Set<String> categorySet = categoryEmbeddingMap.keySet();
        Iterator<String> itr = categorySet.iterator();
        while (itr.hasNext()) {
            String category = itr.next();
            if (!category.equals(queryCategory)) {
                double[] categoryEmbedding = categoryEmbeddingMap.get(category);
                double aa = 0, bb = 0, ab = 0;
                for (int i = 0; i < queryEmbedding.length; i++) {
                    ab += queryEmbedding[i] * categoryEmbedding[i];
                    aa += queryEmbedding[i] * queryEmbedding[i];
                    bb += categoryEmbedding[i] * categoryEmbedding[i];
                }
                double cosSim = ab / (Math.sqrt(aa) * Math.sqrt(bb));
                rankList.add(category + "@" + cosSim);
            }
        }

        Collections.sort(rankList, new SortByProbability());

        for (int i = 0; i < topK.length; i++) {
            int temptopK = topK[i];
            for (int j = 0; j < Math.min(rankList.size(), temptopK); j++) {
                if (rankList.get(j).split("@")[0].equals(topCategory)) {
                    score[i] = 1;
                    break;
                }
            }
        }
        return score;
    }

    private double queryTop1(String queryCategory, HashMap<String, double[]> categoryEmbeddingMap, String topCategory) {
        double score = 0;
        int rank = 0;
        double[] queryEmbedding = categoryEmbeddingMap.get(queryCategory);
        ArrayList<String> rankList = new ArrayList<String>();
        Set<String> categorySet = categoryEmbeddingMap.keySet();
        Iterator<String> itr = categorySet.iterator();
        while (itr.hasNext()) {
            String category = itr.next();
            if (!category.equals(queryCategory)) {
                double[] categoryEmbedding = categoryEmbeddingMap.get(category);
                double aa = 0, bb = 0, ab = 0;
                for (int i = 0; i < queryEmbedding.length; i++) {
                    ab += queryEmbedding[i] * categoryEmbedding[i];
                    aa += queryEmbedding[i] * queryEmbedding[i];
                    bb += categoryEmbedding[i] * categoryEmbedding[i];
                }
                double cosSim = ab / (Math.sqrt(aa) * Math.sqrt(bb));
//				double cosSim = ab;
                rankList.add(category + "@" + cosSim);
            }
        }

        Collections.sort(rankList, new SortByProbability());

        for (int i = 0; i < rankList.size(); i++) {
            if (rankList.get(i).split("@")[0].equals(topCategory)) {
                rank = i + 1;
                break;
            }
        }

        score = 1.0 / rank;
        return score;
    }

    private double queryTop1Random(ArrayList<String> categoryList, String topCategory) {
        double score = 0;
        int rank = 0;
        Collections.shuffle(categoryList);
        for (int i = 0; i < categoryList.size(); i++) {
            if (categoryList.get(i).equals(topCategory)) {
                rank = i + 1;
                break;
            }
        }

        score = 1.0 / rank;
        return score;
    }

}

class SortByProbability implements Comparator<String> {
    @Override
    public int compare(String o1, String o2) {
        double prob1 = Double.parseDouble(o1.split("@")[1]);
        double prob2 = Double.parseDouble(o2.split("@")[1]);
        if (prob1 >= prob2) {
            return -1;
        } else {
            return 1;
        }
    }

}
