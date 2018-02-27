
/**
 * Implement the Lesk algorithm for Word Sense Disambiguation (WSD)
 */
import java.util.*;
import java.util.Map.Entry;

import com.sun.javafx.scene.traversal.Algorithm;

import java.io.*;
import javafx.util.Pair;

import edu.mit.jwi.*;
import edu.mit.jwi.Dictionary;
import edu.mit.jwi.item.*;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.logging.RedwoodConfiguration;

public class Lesk {

	/**
	 * Each entry is a sentence where there is at least a word to be disambiguate.
	 * E.g., testCorpus.get(0) is Sentence object representing "It is a full scale,
	 * small, but efficient house that can become a year' round retreat complete in
	 * every detail."
	 **/
	private ArrayList<Sentence> testCorpus = new ArrayList<Sentence>();

	/**
	 * Each entry is a list of locations (integers) where a word needs to be
	 * disambiguate. The index here is in accordance to testCorpus. E.g.,
	 * ambiguousLocations.get(0) is a list [13] ambiguousLocations.get(1) is a list
	 * [10, 28]
	 **/
	private ArrayList<ArrayList<Integer>> ambiguousLocations = new ArrayList<ArrayList<Integer>>();

	/**
	 * Each entry is a list of pairs, where each pair is the lemma and POS tag of an
	 * ambiguous word. E.g., ambiguousWords.get(0) is [(become, VERB)]
	 * ambiguousWords.get(1) is [(take, VERB), (apply, VERB)]
	 */
	private ArrayList<ArrayList<Pair<String, String>>> ambiguousWords = new ArrayList<ArrayList<Pair<String, String>>>();

	/**
	 * Each entry is a list of maps, each of which maps from a sense key to
	 * similarity(context, signature) E.g., predictions.get(1) = [{take%2:30:01:: ->
	 * 0.9, take%2:38:09:: -> 0.1}, {apply%2:40:00:: -> 0.1}]
	 */
	private ArrayList<ArrayList<HashMap<String, Double>>> predictions = new ArrayList<ArrayList<HashMap<String, Double>>>();

	/**
	 * Each entry is a list of ground truth senses for the ambiguous locations. Each
	 * String object can contain multiple synset ids, separated by comma. E.g.,
	 * groundTruths.get(0) is a list of strings
	 * ["become%2:30:00::,become%2:42:01::"] groundTruths.get(1) is a list of
	 * strings
	 * ["take%2:30:01::,take%2:38:09::,take%2:38:10::,take%2:38:11::,take%2:42:10::",
	 * "apply%2:40:00::"]
	 */
	private ArrayList<ArrayList<String>> groundTruths = new ArrayList<ArrayList<String>>();

	/* This section contains the NLP tools */

	private Set<String> POS = new HashSet<String>(Arrays.asList("ADJECTIVE", "ADVERB", "NOUN", "VERB"));

	private static final Set<String> context_option = new HashSet<String>(Arrays.asList("ALL_WORDS", "ALL_WORDS_R", "WINDOW", "POS"));
	
	private static final Set<String> sim_opt = new HashSet<String>(Arrays.asList("JACCARD", "COSINE"));

	private IDictionary wordnetdict;

	private StanfordCoreNLP pipeline;

	private Set<String> stopwords;

	private Set<String> POSTags;
	
	private ArrayList<Integer> fileS;

	public Lesk() throws FileNotFoundException {
		// initialize stopwords
		stopwords = new HashSet<>();
		File file = new File("data/stopwords.txt");
		FileReader fr = new FileReader(file);
		BufferedReader br = new BufferedReader(fr);
		String line = null;
		try {
			while ((line = br.readLine()) != null) {
				stopwords.add(line);
			}
			br.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// initialize wordnetdict
		wordnetdict = new edu.mit.jwi.Dictionary(new File("data/dict/"));
		try {
			wordnetdict.open();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// initialize pipeline
		// creates a StanfordCoreNLP object, with POS tagging, lemmatization, NER,
		// parsing, and coreference resolution
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
		pipeline = new StanfordCoreNLP(props);

		// initialize POSTags
		POSTags = new HashSet<String>() {
			{
				add("NN");
				add("VB");
				add("JJ");
				add("RB");
			}
		};

	}

	/**
	 * Convert a pos tag in the input file to a POS tag that WordNet can recognize
	 * (JWI needs this). We only handle adjectives, adverbs, nouns and verbs.
	 * 
	 * @param pos:
	 *            a POS tag from an input file.
	 * @return JWI POS tag.
	 */
	private String toJwiPOS(String pos) {
		if (pos.equals("ADJ")) {
			return "ADJECTIVE";
		} else if (pos.equals("ADV")) {
			return "ADVERB";
		} else if (pos.equals("NOUN") || pos.equals("VERB")) {
			return pos;
		} else {
			return null;
		}
	}

	/**
	 * This function fills up testCorpus, ambiguousLocations and groundTruths lists
	 * 
	 * @param filename
	 */
	public void readTestData(String filename) throws Exception {
		File dir = new File(filename);
		File[] f = dir.listFiles();

		ArrayList<Integer> al = new ArrayList<>();
		ArrayList<Pair<String, String>> aw = new ArrayList<>();
		ArrayList<String> gt = new ArrayList<>();

		fileS = new ArrayList<>();
		int count_sentence = 0;
		for (File file : f) {
			boolean isFirstLine = true;
			String line = null;
			BufferedReader br = new BufferedReader(new FileReader(file));
			while ((line = br.readLine()) != null) {
				if (isFirstLine) {
					isFirstLine = false;
					continue;
				}
				ArrayList<Integer> temp1 = new ArrayList<>();
				ArrayList<Pair<String, String>> temp2 = new ArrayList<>();
				ArrayList<String> temp3 = new ArrayList<>();
				String[] str = line.split(" ");
				if (!str[0].matches(".?\\d+")) {
					Sentence s = new Sentence();
					count_sentence++;
					if (al.size() != 0) {
						temp1.addAll(al);
						temp2.addAll(aw);
						temp3.addAll(gt);
						ambiguousLocations.add(temp1);
						ambiguousWords.add(temp2);
						groundTruths.add(temp3);
					}
					al.clear();
					aw.clear();
					gt.clear();
					for (String w : str) {
						Word word = new Word(w);
						s.addWord(word);
					}
					testCorpus.add(s);
				}
				if (str[0].contains("#")) {
					str[0] = str[0].replaceAll("#", "");
					al.add(Integer.parseInt(str[0]));
					aw.add(new Pair(str[1], str[2]));
					gt.add(str[3]);
				}
			}
			fileS.add(count_sentence);
			br.close();
		}
		ArrayList<Integer> temp1 = new ArrayList<>();
		ArrayList<Pair<String, String>> temp2 = new ArrayList<>();
		ArrayList<String> temp3 = new ArrayList<>();
		temp1.addAll(al);
		temp2.addAll(aw);
		temp3.addAll(gt);
		ambiguousLocations.add(temp1);
		ambiguousWords.add(temp2);
		groundTruths.add(temp3);
		
		 //System.out.println(groundTruths);
	}

	/**
	 * Create signatures of the senses of a pos-tagged word.
	 * 
	 * 1. use lemma and pos to look up IIndexWord using Dictionary.getIndexWord() 2.
	 * use IIndexWord.getWordIDs() to find a list of word ids pertaining to this
	 * (lemma, pos) combination. 3. Each word id identifies a sense/synset in
	 * WordNet: use Dictionary's getWord() to find IWord 4. Use the getSynset() api
	 * of IWord to find ISynset Use the getSenseKey() api of IWord to find ISenseKey
	 * (such as charge%1:04:00::) 5. Use the getGloss() api of the ISynset interface
	 * to get the gloss String 6. Use the
	 * Dictionary.getSenseEntry(ISenseKey).getTagCount() to find the frequencies of
	 * the synset.d
	 * 
	 * @param args
	 *            lemma: word form to be disambiguated pos_name: POS tag of the
	 *            wordform, must be in {ADJECTIVE, ADVERB, NOUN, VERB}.
	 * 
	 */
	private Map<String, Pair<String, Integer>> getSignatures(String lemma, String pos_name) {

		POS pos = edu.mit.jwi.item.POS.valueOf((pos_name));
//		System.out.println(pos);
		IIndexWord iIndexWord = wordnetdict.getIndexWord(lemma, pos);
		List<IWordID> wordIDs = iIndexWord.getWordIDs();
		Map<String, Pair<String, Integer>> map = new HashMap<String, Pair<String, Integer>>();
		Pair pair;
		for (IWordID iWordID : wordIDs) {
			IWord word = wordnetdict.getWord(iWordID);
			ISynset iSynset = word.getSynset();
			ISenseKey iSenseKey = word.getSenseKey();
			String gloss = iSynset.getGloss();
			int tagCount = wordnetdict.getSenseEntry(iSenseKey).getTagCount();
			pair = new Pair<String, Integer>(gloss, tagCount);
			map.put(iSenseKey.toString(), pair);
		}
		return map;
	}

	/**
	 * Create a bag-of-words representation of a document (a
	 * sentence/phrase/paragraph/etc.)
	 * 
	 * @param str:
	 *            input string
	 * @return a list of strings (words, punctuation, etc.)
	 */
	private ArrayList<String> str2bow(String str) {
		ArrayList<String> bow = new ArrayList<>();
		String[] allWords = str.split(" ");
		for (String word : allWords) {
			if (!stopwords.contains(word)) {
				if (word.matches(".?\\w+.?")) {
					bow.add(word);
				}
			}
		}
		return bow;
	}

	/**
	 * compute similarity between two bags-of-words.
	 * 
	 * @param bag1
	 *            first bag of words
	 * @param bag2
	 *            second bag of words
	 * @param sim_opt
	 *            COSINE or JACCARD similarity
	 * @return similarity score
	 */
	private double similarity(ArrayList<String> bag1, ArrayList<String> bag2, String sim_opt) {
		if (sim_opt.equalsIgnoreCase("COSINE")) {
			Set result = new HashSet<String>();
			// overlap set
			result.addAll(bag1);
			result.removeAll(bag2);
			double sim = result.size();

			return sim / (double) (Math.sqrt(bag1.size()) * (Math.sqrt(bag2.size())));
		}
		if (sim_opt.equalsIgnoreCase("JACCARD")) {
			Set result = new HashSet<String>();
			Set totalResult = new HashSet<String>();
			// overlap set
			result.addAll(bag1);
			result.removeAll(bag2);
			double sim = result.size();

			totalResult.addAll(bag1);
			totalResult.addAll(bag2);
			double diff = totalResult.size();

			return sim / diff;
		}
		return 0.0;
	}

	/**
	 * This is the WSD function that prediction what senses are more likely.
	 * 
	 * @param context_option:
	 *            one of {ALL_WORDS, ALL_WORDS_R, WINDOW, POS}
	 * @param window_size:
	 *            an odd positive integer > 1
	 * @param sim_option:
	 *            one of {COSINE, JACCARD}
	 * @throws Exception
	 */
	public void predict(String context_option, int window_size, String sim_option) throws Exception {
		if(!Lesk.context_option.contains(context_option))
			throw new Exception("Illegal context_option!");
		if(!Lesk.sim_opt.contains(sim_option))
			throw new Exception("Illegal sim_option!");
		for (int i = 0; i < testCorpus.size(); i++) {
			Sentence sentence = testCorpus.get(i);
//			ArrayList<Integer> locationList = ambiguousLocations.get(i);
			ArrayList<Pair<String, String>> wordList = ambiguousWords.get(i);
			ArrayList<HashMap<String, Double>> temp = new ArrayList<>();
			for (Pair<String, String> pair : wordList) {
				String target = pair.getKey();
				ArrayList<String> context = getContext(context_option, window_size, sentence, target);

				// System.out.println(context);

				String pos = toJwiPOS(pair.getValue());
				Map<String, Pair<String, Integer>> signatures = getSignatures(pair.getKey(), pos);

				// System.out.println(signatures);

				HashMap<String, Double> hm = new HashMap<>();
				for (String str : signatures.keySet()) {
					String gloss = signatures.get(str).getKey();
//					System.out.println(str);
					double similarity = similarity(getSignatureContext("ALL_WORDS",window_size,sim_option,gloss), context, sim_option);
					hm.put(str, similarity);
//					System.out.println(gloss+" "+similarity);
				}
				temp.add(hm);
			}
			System.out.println("current position : "+i);
			predictions.add(temp);
		}
	}

	private ArrayList<String> getSignatureContext(String context_option, int window_size, String sim_option, String text) throws Exception {
		
		if(!Lesk.context_option.contains(context_option))
			throw new Exception("Illegal context_option!");
		if(!Lesk.sim_opt.contains(sim_option))
			throw new Exception("Illegal sim_option!");
		if(context_option.equalsIgnoreCase("WINDOW")) {
			throw new Exception("context_option isn't allowed!");
		}
		
		// create an empty Annotation just with the given text
		Annotation document = new Annotation(text);

		// run all Annotators on this text
		pipeline.annotate(document);
		
		// these are all the sentences in this document
		// a CoreMap is essentially a Map that uses class objects as keys and has values
		// with custom types
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);

		ArrayList<String> list = new ArrayList<>();
		Sentence s = new Sentence();
		for (CoreMap sentence : sentences) {
			// traversing the words in the current sentence
			// a CoreLabel is a CoreMap with additional token-specific methods
			for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
				// this is the text of the token
				String word = token.get(TextAnnotation.class);
				// this is the POS tag of the token
				String pos = token.get(PartOfSpeechAnnotation.class);
				s.addWord(new Word(word, pos));
				// this is the NER label of the token
				// String ne = token.get(NamedEntityTagAnnotation.class);
			}

			// this is the parse tree of the current sentence
			// Tree tree = sentence.get(TreeAnnotation.class);
			// System.out.println(tree);

		}
		list = getContext(context_option, window_size, s, null);

		// This is the coreference link graph
		// Each chain stores a set of mentions that link to each other,
		// along with a method for getting the most representative mention
		// Both sentence and token offsets start at 1!
		// Map<Integer, CorefChain> graph = document.get(CorefChainAnnotation.class);

		return list;
	}

	private ArrayList<String> getContext(String context_option, int window_size, Sentence sentence, String target)
			throws Exception {
		ArrayList<String> context = new ArrayList<>();
		if (context_option.equalsIgnoreCase("ALL_WORDS")) {
			context = sentence.getAllWords();
		}
		if (context_option.equalsIgnoreCase("ALL_WORDS_R")) {
			for (Word word : sentence) {
				if (!stopwords.contains(word.getLemme())) {
					context.add(word.getLemme());
				}
			}
		}
		if (context_option.equalsIgnoreCase("WINDOW")) {
			if (window_size / 2 == 0)
				throw new Exception("please give an odd window_size");
			int index = sentence.getAllWords().indexOf(target);
			for (int i = index - window_size / 2 > 0 ? index - window_size / 2 : 0; i <= Math
					.min(index + window_size / 2, sentence.length() - 1); i++) {
				context.add(sentence.getWordAt(i).getLemme());
			}
		}
		if (context_option.equalsIgnoreCase("POS")) {
			for (int i = 0; i < sentence.length() - 1; i++) {
				Word w = sentence.getWordAt(i);
				if (POSTags.contains(w.getPosTag())) {
					context.add(w.getLemme());
				}
			}
		}

		return context;
	}

	/**
	 * Multiple senses are concatenated using comma ",". Separate them out.
	 * 
	 * @param senses
	 * @return
	 */
	private ArrayList<String> parseSenseKeys(String senseStr) {
		ArrayList<String> senses = new ArrayList<String>();
		String[] items = senseStr.split(",");
		for (String item : items) {
			senses.add(item);
		}
		return senses;
	}

	/**
	 * Precision/Recall/F1-score at top K positions
	 * 
	 * @param groundTruths:
	 *            a list of sense id strings, such as [become%2:30:00::,
	 *            become%2:42:01::]
	 * @param predictions:
	 *            a map from sense id strings to the predicted similarity
	 * @param K
	 * @return a list of [top K precision, top K recall, top K F1]
	 */
	private ArrayList<Double> evaluate(ArrayList<String> groundTruths, HashMap<String, Double> predictions, int K) {
		
		// descending comparator
	    Comparator<Map.Entry<String, Double>> valueComparator = new Comparator<Map.Entry<String,Double>>() {
	        @Override
	        public int compare(Entry<String, Double> o1,
	                Entry<String, Double> o2) {
	            // TODO Auto-generated method stub
	            return (int) ( o2.getValue()-o1.getValue());
	        }
	    };
	    // map transform to list
	    List<Map.Entry<String, Double>> list = new ArrayList<Map.Entry<String,Double>>(predictions.entrySet());
	    // sort
	    Collections.sort(list,valueComparator);
	    Map<String, Double> map = new HashMap<>();
	    for (Entry<String, Double> entry : list) {
//	        System.out.println(entry.getKey() + ":" + entry.getValue());
	        map.put(entry.getKey(), entry.getValue());
	    }
	    
		Set<String> hs = map.keySet();
		ArrayList<String> pred = new ArrayList(hs);
		ArrayList<String> sim = new ArrayList<>(K);
		sim.addAll(hs);
		sim.retainAll(groundTruths);
		Double precision;
		if(sim.size()!=0) {
			if(K<=hs.size())
				precision = 1.0;
			else
				precision = (double) (sim.size()/ K);
		}else {
			precision = 0.0;
		}
		Double recall = ((double) sim.size()/ (double)hs.size());
		Double f1 = (2 * precision * recall) / (precision + recall);
	
		
		return new ArrayList<Double>(Arrays.asList(precision, recall, f1));
	}

	/**
	 * Test the prediction performance on all test sentences
	 * 
	 * @param K
	 *            Top-K precision/recall/f1
	 */
	public ArrayList<Double> evaluate(int K) {
		ArrayList<Double> list = new ArrayList<Double>();
		Double precision = 0.0;
		Double recall = 0.0;
		Double f1 = 0.0;
		double count = 0.0;
		int file_count = 0;
		for (int i = 0; i < groundTruths.size(); i++) {
			ArrayList<String> gt = groundTruths.get(i);
			ArrayList<HashMap<String, Double>> arrayList = predictions.get(i);
			for (int j = 0; j < arrayList.size(); j++) {
				String str_gt = gt.get(j);
				HashMap<String, Double> hm = arrayList.get(j);
				ArrayList<String> senseKeys = parseSenseKeys(str_gt);
				ArrayList<Double> evaluate = evaluate(senseKeys, hm, K);
				
				//System.out.println(evaluate);
				count++;
				precision += evaluate.get(0);
				recall += evaluate.get(1);
				f1 += evaluate.get(2);

			}
			if(i==fileS.get(file_count)-1) {
				list.add(precision/count);
				list.add(recall/count);
				list.add(f1/count);
				count=0.0;
				precision = 0.0;
				recall = 0.0;
				f1 = 0.0;
				file_count++;
			}
		} 	
		return list;
	}

	/**
	 * @param args[0]
	 *            file name of a test corpus
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		Lesk model = new Lesk();
		try {
			model.readTestData("data/semcor_txt");
		} catch (Exception e) {
			// System.out.println(args[0]);
			e.printStackTrace();
		}
		String context_opt = "ALL_WORDS_R";
		int window_size = 3;
		String sim_opt = "JACCARD";

		model.predict(context_opt, window_size, sim_opt);
		
		//System.out.println(model.predictions);
		
		ArrayList<Double> res = model.evaluate(1);
//		System.out.print(args[0]);
		for (int i = 0; i < res.size(); i=i+3) {
			System.out.println(res.get(i)+"\t"+res.get(i+1)+"\t"+res.get(i+2));
		}
	}
}
