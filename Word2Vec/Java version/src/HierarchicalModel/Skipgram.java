package HierarchicalModel;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;
import java.util.Set;

public class Skipgram {

    int EXP_TABLE_SIZE = 1000;
    double[] expTable = new double[EXP_TABLE_SIZE];
    int MAX_EXP = 6;


    public static void main(String[] args) throws IOException {

    }

    public void skipgram(ArrayList<String> trainList, int negativeNum, int vecSize, String contextPath,
                         String targetPath, double learnRate) throws IOException {

        HashSet<String> targetSet = new HashSet<String>();
        HashSet<String> contextSet = new HashSet<String>();
        ArrayList<String> targetFrequencyList = computeTargetFrequencyList(trainList, targetSet, contextSet);

        HashMap<String, double[]> contextVecMap = new HashMap<String, double[]>();
        HashMap<String, double[]> targetVecMap = new HashMap<String, double[]>();

        initialize(targetSet, contextSet, vecSize, contextVecMap, targetVecMap);

        trainModel(trainList, negativeNum, contextVecMap, targetVecMap, contextPath, learnRate, targetFrequencyList);

        saveModel(contextVecMap, contextPath);
        saveModel(targetVecMap, targetPath);
    }

    public void skipgramMultipleContext(ArrayList<String> trainList, int negativeNum, int vecSize, String contextPath,
                                        String targetPath, double learnRate, double weight) throws IOException {
        HashSet<String> targetSet = new HashSet<String>();
        HashSet<String> contextSet = new HashSet<String>();
        ArrayList<String> targetFrequencyList = computeTargetFrequencyList(trainList, targetSet, contextSet);

        HashMap<String, double[]> contextVecMap = new HashMap<String, double[]>();
        HashMap<String, double[]> targetVecMap = new HashMap<String, double[]>();
        initialize(targetSet, contextSet, vecSize, contextVecMap, targetVecMap);

        trainModelWeight(trainList, negativeNum, contextVecMap, targetVecMap, contextPath, learnRate, weight, targetFrequencyList);

        saveModel(contextVecMap, contextPath);
        saveModel(targetVecMap, targetPath);
    }

    private void initialize(HashSet<String> targetSet, HashSet<String> contextSet, int vecSize, HashMap<String, double[]> contextVecMap,
                            HashMap<String, double[]> targetVecMap) {
        createExpTable();
        Iterator<String> itr = targetSet.iterator();
        while (itr.hasNext()) {
            String target = itr.next();
            double[] vec1 = new double[vecSize];
            Random random = new Random();
            for (int i = 0; i < vec1.length; i++) {
//				vec1[i] = (Math.random()-0.5)/vecSize;
                vec1[i] = random.nextGaussian() * 0.01;
            }
            targetVecMap.put(target, vec1);
        }

        itr = contextSet.iterator();
        while (itr.hasNext()) {
            String context = itr.next();
            double[] vec1 = new double[vecSize];
            Random random = new Random();
            for (int i = 0; i < vec1.length; i++) {
//				vec1[i] = (Math.random()-0.5)/vecSize;
                vec1[i] = random.nextGaussian() * 0.01;
            }
            contextVecMap.put(context, vec1);
        }
    }

    private void trainModel(ArrayList<String> trainList, int negativeNum, HashMap<String, double[]> contextVecMap,
                            HashMap<String, double[]> targetVecMap, String contextPath, double learnRate, ArrayList<String> targetFrequencyList) throws IOException {
        int itrNum = 0;
        double runtime = 0;
        while (true) {
            System.out.println("The " + (itrNum + 1) + "th iteration starts.");
            Collections.shuffle(trainList);
            System.out.println("Training data finishes shuffling.");
            double time1 = System.currentTimeMillis();
            double loss = 0;
            for (int i = 0; i < trainList.size(); i++) {
                String target = trainList.get(i).split(",")[0];
                String[] contextTemp = trainList.get(i).split(",")[1].split("#");
                for (int j = 0; j < contextTemp.length; j++) {
                    String context = contextTemp[j];
                    double[] contextVec = contextVecMap.get(context);//V(w)
                    double[] e = new double[contextVec.length];
                    //positive target
                    double[] postiveTargetVec = targetVecMap.get(target); //theta_u
                    double VwThetau = 0;
                    for (int k = 0; k < postiveTargetVec.length; k++) {
                        VwThetau += postiveTargetVec[k] * contextVec[k];
                    }
                    double q = getSigmoid(VwThetau);

                    double g = learnRate * (1 - q);

                    for (int m = 0; m < e.length; m++) {
                        e[m] += g * postiveTargetVec[m];
                    }

                    for (int m = 0; m < postiveTargetVec.length; m++) {
                        postiveTargetVec[m] += g * contextVec[m];
                    }
                    loss += Math.log(q);

                    String negative = getNegativeCategory(negativeNum, target, targetFrequencyList);
                    String[] negativeTargetTemp = negative.split("#");
                    //negative targets
                    for (int k = 0; k < negativeTargetTemp.length; k++) {
                        String negativeTarget = negativeTargetTemp[k];
                        double[] negativeTargetVec = targetVecMap.get(negativeTarget); //theta_u

                        double VwThetau1 = 0;
                        for (int m = 0; m < negativeTargetVec.length; m++) {
                            VwThetau1 += negativeTargetVec[m] * contextVec[m];
                        }
                        double q1 = getSigmoid(VwThetau1);

                        double g1 = learnRate * (0 - q1);

                        for (int m = 0; m < e.length; m++) {
                            e[m] += g1 * negativeTargetVec[m];
                        }

                        for (int m = 0; m < negativeTargetVec.length; m++) {
                            negativeTargetVec[m] += g1 * contextVec[m];
                        }

                        loss += Math.log(1 - q1);
                    }
                    for (int m = 0; m < e.length; m++) {
                        contextVec[m] += e[m];
                    }
                }
            }
            System.out.println("loss=" + loss);
            itrNum++;
            if (itrNum > 20) {
                break;
            }
            double time2 = System.currentTimeMillis();
            runtime = (time2 - time1) / 1000;
            System.out.println(runtime + "seconds");

            if (itrNum % 2 == 0) {
                learnRate /= 2;
            }
            if (learnRate < 0.0000025) {
                learnRate = 0.0000025;
            }
//			if(itrNum%2==0) {
//				saveModel(contextVecMap, "C:\\Users\\chenm\\Desktop\\input\\temp\\embedding"+itrNum+".csv");
//			}
        }
    }


    private void trainModelWeight(ArrayList<String> trainList, int negativeNum, HashMap<String, double[]> contextVecMap,
                                  HashMap<String, double[]> targetVecMap, String contextPath, double learnRate, double weight, ArrayList<String> targetFrequencyList) throws IOException {
        int itrNum = 0;
        double runtime = 0;
        while (true) {
            System.out.println("The " + (itrNum + 1) + "th iteration starts.");
            Collections.shuffle(trainList);
            System.out.println("Training data finishes shuffling.");
            double time1 = System.currentTimeMillis();
            double loss = 0;
            for (int i = 0; i < trainList.size(); i++) {
                String target = trainList.get(i).split(",")[0];
                String firstContext = trainList.get(i).split(",")[1];
                String secondContext = trainList.get(i).split(",")[2];

                String[] contextTemp1 = firstContext.split("#");
                for (int j = 0; j < contextTemp1.length; j++) {
                    String context = contextTemp1[j];
                    double[] contextVec = contextVecMap.get(context);//V(w)
                    double[] e = new double[contextVec.length];
                    //positive target
                    double[] postiveTargetVec = targetVecMap.get(target); //theta_u
                    double VwThetau = 0;
                    for (int k = 0; k < postiveTargetVec.length; k++) {
                        VwThetau += postiveTargetVec[k] * contextVec[k];
                    }
                    double q = getSigmoid(VwThetau);

                    double g = learnRate * (1 - q);

                    for (int m = 0; m < e.length; m++) {
                        e[m] += g * postiveTargetVec[m];
                    }

                    for (int m = 0; m < postiveTargetVec.length; m++) {
                        postiveTargetVec[m] += g * contextVec[m] * weight;

                    }
                    loss += Math.log(q) * weight;

                    String negative = getNegativeCategory(negativeNum, target, targetFrequencyList);
                    String[] negativeTargetTemp = negative.split("#");
                    //negative targets
                    for (int k = 0; k < negativeTargetTemp.length; k++) {
                        String negativeTarget = negativeTargetTemp[k];
                        double[] negativeTargetVec = targetVecMap.get(negativeTarget); //theta_u
                        double VwThetau1 = 0;
                        for (int m = 0; m < negativeTargetVec.length; m++) {
                            VwThetau1 += negativeTargetVec[m] * contextVec[m];
                        }
                        double q1 = getSigmoid(VwThetau1);
                        double g1 = learnRate * (0 - q1);

                        for (int m = 0; m < e.length; m++) {
                            e[m] += g1 * negativeTargetVec[m];
                        }

                        for (int m = 0; m < negativeTargetVec.length; m++) {
                            negativeTargetVec[m] += g1 * contextVec[m] * weight;
                        }

                        loss += Math.log(1 - q1) * weight;
                    }

                    for (int m = 0; m < e.length; m++) {
                        contextVec[m] += e[m] * weight;
                    }
                }

                String[] contextTemp2 = secondContext.split("#");
                for (int j = 0; j < contextTemp2.length; j++) {
                    String context = contextTemp2[j];
                    double[] contextVec = contextVecMap.get(context);//V(w)
                    double[] e = new double[contextVec.length];
                    //positive target
                    double[] postiveTargetVec = targetVecMap.get(target); //theta_u
                    double VwThetau = 0;
                    for (int k = 0; k < postiveTargetVec.length; k++) {
                        VwThetau += postiveTargetVec[k] * contextVec[k];
                    }
                    double q = getSigmoid(VwThetau);

                    double g = learnRate * (1 - q);

                    for (int m = 0; m < e.length; m++) {
                        e[m] += g * postiveTargetVec[m];
                    }

                    for (int m = 0; m < postiveTargetVec.length; m++) {
                        postiveTargetVec[m] += g * contextVec[m] * (1 - weight);
                    }

                    loss += Math.log(q) * (1 - weight);

                    String negative = getNegativeCategory(negativeNum, target, targetFrequencyList);
                    String[] negativeTargetTemp = negative.split("#");
                    //negative targets
                    for (int k = 0; k < negativeTargetTemp.length; k++) {
                        String negativeTarget = negativeTargetTemp[k];
                        double[] negativeTargetVec = targetVecMap.get(negativeTarget); //theta_u
                        double VwThetau1 = 0;
                        for (int m = 0; m < negativeTargetVec.length; m++) {
                            VwThetau1 += negativeTargetVec[m] * contextVec[m];
                        }
                        double q1 = getSigmoid(VwThetau1);
                        double g1 = learnRate * (0 - q1);

                        for (int m = 0; m < e.length; m++) {
                            e[m] += g1 * negativeTargetVec[m];
                        }

                        for (int m = 0; m < negativeTargetVec.length; m++) {
                            negativeTargetVec[m] += g1 * contextVec[m] * (1 - weight);
                        }

                        loss += Math.log(1 - q1) * (1 - weight);
                    }

                    for (int m = 0; m < e.length; m++) {
                        contextVec[m] += e[m] * (1 - weight);
                    }
                }

            }

            System.out.println("loss=" + loss);
            itrNum++;
            if (itrNum > 20) {
                break;
            }
            double time2 = System.currentTimeMillis();
            runtime = (time2 - time1) / 1000;
            System.out.println(runtime + "seconds");
            if (itrNum % 2 == 0) {
                learnRate /= 2;
            }
            if (learnRate < 0.0000025) {
                learnRate = 0.0000025;
            }

//			if(itrNum%5==0) {
//				saveModel(contextVecMap, "C:\\Users\\chenm\\Desktop\\input\\temp\\embedding"+itrNum+".csv");
//			}

//			evaluateModel(contextPath);
        }
    }


    // compute category list according to the frequency
    private ArrayList<String> computeTargetFrequencyList(ArrayList<String> trainList, HashSet<String> targetSet, HashSet<String> contextSet) throws IOException {
        ArrayList<String> targetFrequencyList = new ArrayList<>();
        HashMap<String, Integer> candidateTargetCountMap = new HashMap<String, Integer>();
        for (int i = 0; i < trainList.size(); i++) {
            String target = trainList.get(i).split(",")[0];
            if (candidateTargetCountMap.containsKey(target)) {
                candidateTargetCountMap.put(target, candidateTargetCountMap.get(target) + 1);
            } else {
                candidateTargetCountMap.put(target, 1);
            }
            targetSet.add(target);

            String firstContext = trainList.get(i).split(",")[1];
//				String secondContext = trainList.get(i).split(",")[2];

            String[] temp = firstContext.split("#");
            for (int j = 0; j < temp.length; j++) {
                contextSet.add(temp[j]);
            }

//				String[] temp1 = secondContext.split("#");
//				for(int j=0;j<temp1.length;j++) {
//					contextSet.add(temp1[j]);
//				}
        }

        // normalization
        Set<String> set = candidateTargetCountMap.keySet();
        Iterator<String> itr = set.iterator();
        int min = 10000;
        while (itr.hasNext()) {
            String target = itr.next();
            int count = candidateTargetCountMap.get(target);
            if (count < min) {
                min = count;
            }
        }

        // num of category is related to the frequency's 3/4 cifang
        itr = set.iterator();
        while (itr.hasNext()) {
            String target = itr.next();
            int count = candidateTargetCountMap.get(target);
            int newcount = (int) Math.pow(count / (min + 0.0), 3 / 4);
            for (int i = 0; i < newcount; i++) {
                targetFrequencyList.add(target);
            }
        }
        return targetFrequencyList;
    }

    // sample negative category for a query category
    private String getNegativeCategory(int sampleNum, String queryCategory, ArrayList<String> targetFrequencyList) {
        // negative target category
        String negative = "";
        ArrayList<String> negativeCategoryList = new ArrayList<String>();
        int sampleCount = 0, count = 0;
        while (true) {
            count++;
            int index = (int) Math.min(targetFrequencyList.size() - 1, Math.random() * targetFrequencyList.size());
            String sampleTargetCategory = targetFrequencyList.get(index);
            if (!sampleTargetCategory.equals(queryCategory)) {
                negativeCategoryList.add(sampleTargetCategory);
                sampleCount++;
            }
            // if sample 50 times do not enough negative samples, stop
            if (sampleCount == sampleNum || count > 50) {
                break;
            }
        }

        for (int i = 0; i < negativeCategoryList.size() - 1; i++) {
            negative += negativeCategoryList.get(i) + "#";
        }
        negative += negativeCategoryList.get(negativeCategoryList.size() - 1);
        return negative;
    }

    private void saveModel(HashMap<String, double[]> contextCategoryVecMap1, String path) throws IOException {
        PrintWriter pw = new PrintWriter(new FileWriter(path));
        Set<String> categorySet = contextCategoryVecMap1.keySet();
        Iterator<String> itr = categorySet.iterator();
        while (itr.hasNext()) {
            String category = itr.next();
            pw.print(category);
            double[] vec = contextCategoryVecMap1.get(category);
            for (int i = 0; i < vec.length; i++) {
                pw.print("," + vec[i]);
            }
            pw.println();
        }
        pw.flush();
        pw.close();
    }

    public void getTrainListFromFile(String inputPath, ArrayList<String> trainList) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(inputPath));
        while (true) {
            String line = br.readLine();
            if (line == null)
                break;
            trainList.add(line);
        }
        br.close();
    }

    public void list2File(String inputPath, ArrayList<String> trainList) throws IOException {
        PrintWriter pw = new PrintWriter(new FileWriter(inputPath));
        for (int i = 0; i < trainList.size(); i++) {
            pw.println(trainList.get(i));
        }
        pw.flush();
        pw.close();
    }

    private double getSigmoid(double z) {
        double sigmoidZ = 0;
        // sigmoid
        if (z <= -MAX_EXP)
            sigmoidZ = 0.000001;
        else if (z >= MAX_EXP)
            sigmoidZ = 0.999999;
        else
            sigmoidZ = expTable[(int) ((z + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
        return sigmoidZ;
    }

    /**
     * Precompute the exp() table sigmoid function jinsi jisuan f(x) = x / (x + 1)
     */
    private void createExpTable() {
        for (int i = 0; i < EXP_TABLE_SIZE; i++) {
            expTable[i] = Math.exp(((i / (double) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP));
            expTable[i] = expTable[i] / (expTable[i] + 1);
        }
    }
}
