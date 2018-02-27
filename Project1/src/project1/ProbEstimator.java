package project1;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.DocumentPreprocessor;

public class ProbEstimator {

	//Frequencies of all bigram words in training set
	private HashMap map = new HashMap<String, HashMap<String, Double>>();

	//Frequencies of frequencies
	private HashMap ffMap = new HashMap<Double, Double>();

	//N0
	private Double zeroBigram = 0.0;
	
	//N1+N2+...+Nn
	private Double totalSeenCount = 0.0;
	
	//N
	private int totalCount = 0;

	/**
	 * Write frequencies of all bigram words in file
	 * @param inputFileName
	 * @param outputFileName
	 * @throws Exception
	 */
	private void bigramModel(String inputFileName, String outputFileName)
			throws Exception {
		FileWriter fw = new FileWriter(outputFileName);
		BufferedWriter bw = new BufferedWriter(fw);

		DocumentPreprocessor dp = new DocumentPreprocessor(inputFileName);

		// for each sentence
		for (List<HasWord> sentence : dp) {
			for (int i = 1; i < sentence.size(); i++) {
				HasWord w = sentence.get(i);
				HasWord v = sentence.get(i - 1);
				totalCount++;
				BigramCount(w.word(), v.word());
			}
		}
		Set entrySet = map.entrySet();
		Iterator iterator = entrySet.iterator();
		while (iterator.hasNext()) {
			Entry<String, HashMap<String, Double>> next = (Entry<String, HashMap<String, Double>>) iterator
					.next();
			String w = next.getKey();
			HashMap<String, Double> hm = next.getValue();
			totalSeenCount += (hm.size() - 1);
			// Set entry = hm.entrySet();
			// Iterator it = entry.iterator();
			// double count = next.getValue().get(next.getKey());
			// while (it.hasNext()) {
			// Entry<String, Double> nxt = (Entry<String, Double>) it.next();
			// nxt.setValue(nxt.getValue()/count);
			// }
			bw.write(w + ":" + hm.toString() + "\n");
		}
		zeroBigram = map.size() * map.size() - map.size() - totalSeenCount;
		bw.close();
		fw.close();
	}

	/**
	 * count frequency of (w,v)
	 * add (w,v) and it frequency in map
	 * @param w
	 * @param v
	 */
	private void BigramCount(String w, String v) {

		if (!map.containsKey(w)) { 
			HashMap subMap = new HashMap();
			subMap.put(w, new Double(1.0));
			subMap.put(v, new Double(1.0));
			map.put(w, subMap);
		} else {
			HashMap temp = (HashMap) map.get(w);
			Double count = ((Double) temp.get(w)).doubleValue() + 1.0;
			temp.put(w, new Double(count));
			if (temp.containsKey(v)) {
				Double value = ((Double) temp.get(v)).doubleValue() + 1.0;
				temp.put(v, new Double(value));
			} else {
				temp.put(v, new Double(1.0));
			}
			map.put(w, temp);
		}
	}

	/**
	 * Frequencies of frequencies
	 * @param outputFileName
	 * @throws Exception
	 */
	private void ff(String outputFileName) throws Exception {
		FileWriter fw = new FileWriter(outputFileName);
		BufferedWriter bw = new BufferedWriter(fw);
		bw.write(0 + " " + zeroBigram + "\n");

		for (Double c = 0.0; c <= 7; c++) {
			if (c == 0)
				ffMap.put(c, zeroBigram);
			else {
				ffMap.put(c, new Double(0));
			}
		}

		for (Double c = 1.0; c <=500 ;c++) {
			Double count = 0.0;
			Set entrySet = map.entrySet();
			Iterator iterator = entrySet.iterator();
			while (iterator.hasNext()) {
				Entry<String, HashMap<String, Double>> next = (Entry<String, HashMap<String, Double>>) iterator
						.next();
				Set entry = next.getValue().entrySet();
				Iterator it = entry.iterator();
				while (it.hasNext()) {
					Entry<String, Double> nxt = (Entry<String, Double>) it
							.next();
					if (nxt.getValue().doubleValue() == c) {
						count++;
					}
				}
			}
			ffMap.replace(c, count);
			bw.write(c + " " + count + "\n");
		}
		bw.close();
		fw.close();
	}

	/**
	 * Use Good-Turning Smoothing technique handle with unseen tokens
	 * @param outputFileName
	 * @throws IOException
	 */
	private void gtTable(String outputFileName) throws IOException {
		FileWriter fw = new FileWriter(outputFileName);
		BufferedWriter bw = new BufferedWriter(fw);
		Set es = ffMap.entrySet();
		Iterator i = es.iterator();
		Double c1 = 0.0;
		while (i.hasNext()) {
			Entry<Double, Double> next = (Entry<Double, Double>) i.next();
			if (ffMap.get(next.getKey() + 1) != null)
				c1 = (next.getKey() + 1)
						* ((double) ffMap.get(next.getKey() + 1) / next
								.getValue());
			bw.write("c:" + next.getKey() + "  c*:" + c1 + "\n");
		}
		bw.close();
		fw.close();
	}

	/**
	 * Use Laplacian Smoothing technique handle with unseen tokens
	 * @param outputFileName
	 * @throws IOException
	 */
	private void Lap(String outputFileName) throws IOException {
		FileWriter fw = new FileWriter(outputFileName);
		BufferedWriter bw = new BufferedWriter(fw);
		Set es = ffMap.entrySet();
		Iterator it = es.iterator();
		double c1 = 0.0;
		while (it.hasNext()) {
			Entry<Double, Double> next = (Entry<Double, Double>) it.next();
			c1 = (next.getKey() + 1)*totalCount/ (map.size() + totalCount);
			bw.write("c:" + next.getKey() + "  c*:" + c1 + "\n");
		}
		bw.close();
		fw.close();
	}

	public static void main(String[] args) throws Exception {
		ProbEstimator probEstimator = new ProbEstimator();
		probEstimator
				.bigramModel("data/train_token.txt", "results/bigrams.txt");
		// probEstimator.bigramModel("data/test_tokens_fake.txt",
		// "results/bigrams2.txt");
		probEstimator.ff("results/ff.txt");
		probEstimator.gtTable("results/GTTable.txt");
		probEstimator.Lap("results/Laplacian.txt");
		//System.out.println(probEstimator.totalCount);
	}
}
