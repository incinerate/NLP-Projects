package project1;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import javax.swing.plaf.basic.BasicScrollPaneUI.HSBChangeListener;

import org.junit.Test;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.DocumentPreprocessor;

public class Predictor {

	//Confusing words
	private HashMap cfw = new HashMap<String, String>();

	//Train model
	private HashMap trainMap = new HashMap<String, HashMap<String, Double>>();

	/**
	 * Use training model check Test file, create results
	 * @param inputFileName
	 * @param outputFileName
	 * @param testFileName
	 * @throws Exception
	 */
	public void test_Prediction(String inputFileName, String outputFileName,
			String testFileName) throws Exception {
		FileWriter fw = new FileWriter(outputFileName);
		BufferedWriter bw = new BufferedWriter(fw);

		DocumentPreprocessor dp = new DocumentPreprocessor(testFileName);
		dp.setSentenceDelimiter("</s>");

		BufferedReader br = new BufferedReader(new FileReader(inputFileName));

		String cur_line = null;

		while ((cur_line = br.readLine()) != null) {
			String[] parts = cur_line.split(":");
			cfw.put(parts[0], parts[1]);
			cfw.put(parts[1], parts[0]);
		}
		int rawcount = 0;
		
		List<Integer> location = new ArrayList<Integer>();
		// for each sentence
		boolean flag = true;
		for (List<HasWord> sentence : dp) {
			if(!flag) bw.write(",\n");
			flag = true;
			for (int i = 0; i < sentence.size(); i++) {
				HasWord hw = sentence.get(i);
				if (cfw.containsKey(hw.word())) {
					HasWord preHw = sentence.get(i - 1);
					if(preHw.word().contains(","))
						preHw.setWord("c.mma");
					double p1 = bigramProb(hw.word(), preHw.word());
					String cfwd = (String) cfw.get(hw.word());
					double p2 = bigramProb(cfwd, preHw.word());
					if (p2 > p1) {
						if(flag){
							bw.write(rawcount + ":"+i);
							//bw.write(preHw.word()+"  "+hw.word()+" "+sentence);
							flag = false;
						} else {
							bw.write(","+i);
						}
					}
				}
				// if (cfw.containsValue(hw.word())) {
				// HasWord preHw = sentence.get(i - 1);
				// //System.out.println(preHw.word()+"  "+hw.word());
				// double p1 = bigramProb(hw.word(), preHw.word());
				// String cfwd = (String) cfw.get(hw.word());
				// double p2 = bigramProb(cfwd, preHw.word());
				// bw.write(preHw.word()+"  "+hw.word()+" p1:"+p1+"\n");
				// bw.write(preHw.word()+"  "+cfwd+" p2:"+p2+"\n");
				// }
			}
			rawcount++;
		}
		bw.close();
		fw.close();
	}

	/**
	 * Calculate c*(w,v)
	 * @param word
	 * @param preWord
	 * @return
	 * @throws Exception
	 */
	private double bigramProb(String word, String preWord) throws Exception {
		// TODO Auto-generated method stub
		double p = 0;

		BufferedReader br = new BufferedReader(new FileReader(
				"results/bigrams.txt"));
		BufferedReader br1 = new BufferedReader(new FileReader(
				"results/GTTable.txt"));

		String line = null;
		String cur_line = null;
		Double c1;

		Map<String, Double> subMap;
		while ((cur_line = br.readLine()) != null) {
			String[] str = cur_line.split(":");
			subMap = mapStringToMap(str[1]);
			trainMap.put(str[0], subMap);
		}

		Map<String, Double> hm = (Map<String, Double>) trainMap.get(word);

		if (trainMap.containsKey(word) && hm.containsKey(preWord)) {
			c1 = hm.get(preWord);
		} else {
			c1 = 0.0;
		}

		while ((line = br1.readLine()) != null) {
			if (c1.doubleValue() == Double.valueOf((line.split(" ")[0]
					.split(":")[1]))) {
				p = Double.valueOf(line.split(":")[2]);
			} 
		}
		if(p==0)
			p = c1.doubleValue();
		return p;
	}

	/**
	 * Used for read .txt file created in last .java file
	 * @param str
	 * @return
	 */
	public static Map<String, Double> mapStringToMap(String str) {
		str = str.substring(1, str.length() - 1);
		str = str.replaceAll(" ", "");
		String[] strs = str.split(",");
		Map<String, Double> map = new HashMap<String, Double>();
		for (String string : strs) {
			String key = string.split("=")[0];
			String[] subStr = string.split("=");
			Double value = Double.valueOf(subStr[subStr.length - 1]);
			map.put(key, value);
		}
		return map;
	}

	public static void main(String[] args) throws Exception {
		Predictor predictor = new Predictor();
		predictor.test_Prediction("data/all_confusingWords.txt",
				"results/test_prediction.txt", "data/test_tokens_fake.txt");

	}
}
