import java.util.Arrays;
import java.util.Hashtable;
import java.util.ArrayList;
import java.util.List;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import Jama.Matrix;

class HMM {
	/* Section for variables regarding the data */

	//
	private ArrayList<Sentence> labeled_corpus;

	//
	private ArrayList<Sentence> unlabeled_corpus;

	// number of pos tags
	int num_postags;

	// number of train words
	int num_train;

	// mapping POS tags in String to their indices
	Hashtable<String, Integer> pos_tags;

	// inverse of pos_tags: mapping POS tag indices to their String format
	Hashtable<Integer, String> inv_pos_tags;

	// vocabulary size
	int num_words;

	Hashtable<String, Integer> vocabulary;

	private int max_sentence_length;

	/* Section for variables in HMM */

	// transition matrix
	private Matrix A;
	// transition matrix of test data
	private Matrix A0;

	// emission matrix
	private Matrix B;
	// emission matrix of test data
	private Matrix B0;

	// prior of pos tags
	private Matrix pi;
	// prior of pos tags in test data
	private Matrix pi0;

	// store the scaled alpha and beta
	private Matrix alpha;

	private Matrix beta;

	// scales to prevent alpha and beta from underflowing
	private Matrix scales;

	// logged v for Viterbi
	private Matrix v;
	private Matrix back_pointer;
	private Matrix pred_seq;

	// \xi_t(i): expected frequency of pos tag i at position t. Use as an
	// accumulator.
	private Matrix gamma;

	// \xi_t(i, j): expected frequency of transiting from pos tag i to j at
	// position t. Use as an accumulator.
	private Matrix digamma;

	// \xi_t(i,w): expected frequency of pos tag i emits word w.
	private Matrix gamma_w;

	// \xi_0(i): expected frequency of pos tag i at position 0.
	private Matrix gamma_0;

	/* Section of parameters for running the algorithms */

	// smoothing epsilon for the B matrix (since there are likely to be unseen
	// words in the training corpus)
	// preventing B(j, o) from being 0
	private double smoothing_eps = 0.1;

	// number of iterations of EM
	private int max_iters = 30;

	// \mu: a value in [0,1] to balance estimations from MLE and EM
	// \mu=1: totally supervised and \mu = 0: use MLE to start but then use EM
	// totally.
	private double mu = 0.9;

	/* Section of variables monitoring training */

	// record the changes in log likelihood during EM
	private double[] log_likelihood = new double[max_iters];

	/**
	 * Constructor with input corpora. Set up the basic statistics of the
	 * corpora.
	 */
	public HMM(ArrayList<Sentence> _labeled_corpus,
			ArrayList<Sentence> _unlabeled_corpus) {
		this.labeled_corpus = _labeled_corpus;
		this.unlabeled_corpus = _unlabeled_corpus;

	}

	/**
	 * Set the semi-supervised parameter \mu
	 */
	public void setMu(double _mu) {
		if (_mu < 0) {
			this.mu = 0.0;
		} else if (_mu > 1) {
			this.mu = 1.0;
		}
		this.mu = _mu;
	}

	/**
	 * Create HMM variables.
	 */
	public void prepareMatrices() {
		pos_tags = new Hashtable<String, Integer>();
		vocabulary = new Hashtable<String, Integer>();
		int i = 0, j = 0;
		for (Sentence sentence : labeled_corpus) {
			max_sentence_length = Math.max(max_sentence_length,
					sentence.length());
			for (Word word : sentence) {
				if (!pos_tags.containsKey(word.getPosTag())) {
					pos_tags.put(word.getPosTag(), i);
					i++;
				}
				if (!vocabulary.containsKey(word.getLemme())) {
					vocabulary.put(word.getLemme(), j);
					j++;
				}
			}
		}

		// reverse pos_tags
		inv_pos_tags = new Hashtable<Integer, String>();
		for (String key : pos_tags.keySet()) {
			inv_pos_tags.put(pos_tags.get(key), key);
		}

		int j0 = 0, i0 = 0;
		for (Sentence sentence : unlabeled_corpus) {
			max_sentence_length = Math.max(max_sentence_length,
					sentence.length());
			for (Word word : sentence) {
				if (!pos_tags.containsKey(word.getPosTag())) {
					pos_tags.put(word.getPosTag(), i);
					i0++;
				}
				if (!vocabulary.containsKey(word.getLemme())) {
					vocabulary.put(word.getLemme(), j0);
					j0++;
				}
			}
		}
		num_postags = pos_tags.size();
		num_words = vocabulary.size();
		A = new Matrix(new double[num_postags][num_postags + 1]);
		B = new Matrix(new double[num_postags][num_words]);
		pi = new Matrix(new double[1][num_postags]);
		A0 = new Matrix(new double[num_postags][num_postags + 1]);
		B0 = new Matrix(new double[num_postags][num_words]);
		pi0 = new Matrix(new double[1][num_postags]);
	}

	/**
	 * MLE A, B and pi on a labeled corpus used as initialization of the
	 * parameters.
	 * 
	 * @throws IOException
	 */
	public void mle() {
		int[] count_tags = new int[num_postags];
		for (Sentence sentence : labeled_corpus) {
			int prePostag_id = -1;
			for (int i = 0; i < sentence.length(); i++) {
				Word word = sentence.getWordAt(i);
				int posTag_id = pos_tags.get(word.getPosTag()).intValue();
				int lemme_id = vocabulary.get(word.getLemme()).intValue();
				count_tags[posTag_id]++;
				if (i == 0) {
					pi0.set(0, posTag_id, pi0.get(0, posTag_id) + 1);
					pi.set(0, posTag_id, pi.get(0, posTag_id) + 1);
				} else {
					A0.set(prePostag_id, posTag_id,
							A0.get(prePostag_id, posTag_id) + 1);
					A.set(prePostag_id, posTag_id,
							A.get(prePostag_id, posTag_id) + 1);
				}
				B0.set(posTag_id, lemme_id, B0.get(posTag_id, lemme_id) + 1);
				B.set(posTag_id, lemme_id, B.get(posTag_id, lemme_id) + 1);
				prePostag_id = posTag_id;
				if (i == sentence.length() - 1) {
					A0.set(posTag_id, num_postags,
							A0.get(posTag_id, num_postags) + 1);
					A.set(posTag_id, num_postags,
							A.get(posTag_id, num_postags) + 1);
				}
			}
		}

		for (int i = 0; i < num_postags; ++i) {
			for (int j = 0; j < num_words; ++j) {
				B0.set(i, j, B0.get(i, j) + smoothing_eps);
				B.set(i, j, B.get(i, j) + smoothing_eps);
			}
		}

		// normalize
		normalization(A);
		normalization(B);
		normalization(A0);
		normalization(B0);
		normalization(pi);
		normalization(pi0);

		// A = mergeMatrix(mu, A, A0);
		// B = mergeMatrix(mu, B, B0);
		// pi = mergeMatrix(mu, pi, pi0);
	}

	/**
	 * normalize function
	 * 
	 * @param m
	 */
	private void normalization(Matrix m) {
		for (int i = 0; i < m.getRowDimension(); i++) {
			double sum = 0;
			for (int j = 0; j < m.getColumnDimension(); j++) {
				sum += m.get(i, j);
			}
			for (int j = 0; j < m.getColumnDimension(); j++) {
				m.set(i, j, m.get(i, j) / sum);
			}
		}
	}

	/**
	 * merge HMM_train{A,B,PI} and HMM_test{A0,B0,PI0}
	 * 
	 * @return Matrix
	 */
	private Matrix mergeMatrix(double mu, Matrix first, Matrix second) {

		return first.times(mu).plus(second.times(1 - mu));

	}

	/**
	 * Main EM algorithm.
	 */
	public void em() {

		alpha = new Matrix(num_postags, max_sentence_length);
		beta = new Matrix(num_postags, max_sentence_length);
		scales = new Matrix(1, max_sentence_length);

		digamma = new Matrix(num_postags, num_postags + 1);

		// gamma = new Matrix(new double[1][num_postags]);
		gamma_0 = new Matrix(1, num_postags);
		gamma_w = new Matrix(num_postags, num_words);
		

		for (int i = 0; i < max_iters; i++) {
			double P = 0;
			for (int j = 0; j < unlabeled_corpus.size(); j++) {
				double Prob = expection(unlabeled_corpus.get(j));
				P+=Prob;
			}

			log_likelihood[i] += P;

			maximization();

			A = mergeMatrix(mu, A0, A);
			B = mergeMatrix(mu, B0, B);
			pi = mergeMatrix(mu, pi0, pi);
		                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        	
//			for(int k = 0; k < num_postags; k++){
//                for(int j = 0; j < num_postags + 1; j++) {
//                    A.set(k, j, mu * A0.get(k, j) + (1-mu) * A.get(k, j));
//                }
//
//                for(int j = 0; j < num_words; ++j) {
//                    B.set(k, j, mu * B0.get(k, j) + (1-mu) * B.get(k, j));
//                }
//            }
//        }
//    
//
//
//        for(int tag_id = 0; tag_id < num_postags; ++tag_id) {
//            pi.set(0, tag_id, mu * pi0.get(0, tag_id) + (1-mu) * pi.get(0, tag_id));
//        }

			normalization(A);
			normalization(B);
			normalization(pi);
			normalization(A0);
			normalization(B0);
			normalization(pi0);
		}
	}	

	/**
	 * Prediction Find the most likely pos tag for each word of the sentences in
	 * the unlabeled corpus.
	 */
	public double predict() {
		v = new Matrix(num_postags, max_sentence_length);
		back_pointer = new Matrix(num_postags, max_sentence_length);
		pred_seq = new Matrix(unlabeled_corpus.size(), max_sentence_length);
		int correct = 0;
		int all = 0;

		for (int i = 0; i < unlabeled_corpus.size(); i++) {
			Sentence s = unlabeled_corpus.get(i);
			all += s.length();
			int index = (int) viterbi(s);

			int k = s.length() - 1;
			while (k >= 0) {
				pred_seq.set(i, k, index);
				if (index == pos_tags.get(s.getWordAt(k).getPosTag())
						.intValue())
					correct++;
				index = (int) back_pointer.get(index, k);
				k--;
			}
		}
		return (double) correct / all;
	}

	/**
	 * Output prediction
	 */
	public void outputPredictions(String outFileName) throws IOException {
		FileWriter fw = new FileWriter(outFileName);
		BufferedWriter bw = new BufferedWriter(fw);
		int correct = 0;
		int all = 0;
		for (int i = 0; i < unlabeled_corpus.size(); ++i) {
			Sentence s = unlabeled_corpus.get(i);
			for (int j = 0; j < s.length(); j++) {
				bw.write(s.getWordAt(j).getLemme() + " ");
				bw.write(inv_pos_tags.get((int) pred_seq.get(i, j)) + "\n");
				if (s.getWordAt(j).getPosTag()
						.equals(inv_pos_tags.get((int) pred_seq.get(i, j))))
					correct++;
				all++;
			}
			bw.write("\n");
		}
		bw.close();
		fw.close();
	}

	/**
	 * outputTrainingLog
	 * 
	 * @param model
	 */
	public void outputTrainingLog(String outFileName) throws IOException {

		FileWriter fw = new FileWriter(outFileName);
		BufferedWriter bw = new BufferedWriter(fw);

		for (int j = 0; j < this.max_iters; j++) {
			bw.write(this.log_likelihood[j] + "\n");
		}
		bw.flush();
		bw.close();
	}

	/**
	 * Expection step of the EM (Baum-Welch) algorithm for one sentence.
	 * \xi_t(i,j) and \xi_t(i) are computed for a sentence
	 */
	private double expection(Sentence s) {

		// calculation of Forward and Backward Variables from the current model
		double fwd = forward(s);
		double bwd = backwardScale(s);

		double[][] arr0 = new double[num_postags][num_postags + 1];
		double[][] arr1 = new double[max_sentence_length][num_postags];
		digamma = new Matrix(arr0);
		gamma = new Matrix(arr1);
		for (int i = 0; i < s.length(); i++) {
			if (gamma.get(i, pos_tags.get(s.getWordAt(i).getPosTag())) == 0)
				gamma.set(i, pos_tags.get((s.getWordAt(i).getPosTag())), 1);
			else {
				gamma.set(
						i,
						pos_tags.get((s.getWordAt(i).getPosTag())),
						gamma.get(i, pos_tags.get(s.getWordAt(i).getPosTag())) + 1);
			}
			Word word = s.getWordAt(i);
			Word nextWord = (i == s.length() - 1 ? s.getWordAt(i) : s
					.getWordAt(i + 1));
			if (i != s.length() - 1) {
				if (digamma.get(pos_tags.get(word.getPosTag()),
						pos_tags.get(nextWord.getPosTag())) == 0)
					digamma.set(pos_tags.get(word.getPosTag()),
							pos_tags.get(nextWord.getPosTag()), 1);
				else {
					double index = digamma.get(pos_tags.get(word.getPosTag()),
							pos_tags.get(nextWord.getPosTag()));
					digamma.set(pos_tags.get(word.getPosTag()),
							pos_tags.get(nextWord.getPosTag()), index + 1);
				}
			}
		}
		return fwd;
	}

	/**
	 * Maximization step of the EM (Baum-Welch) algorithm. Just reestimate A, B
	 * and pi using gamma and digamma
	 * 
	 * @param pi1
	 * @param a1
	 * @param b1
	 */
	public void maximization() {
		
		for (int i = 0; i < num_postags; i++) {

			double scale = smoothing_eps * (num_postags + 1);
			for (int j = 0; j < num_postags + 1; j++) {
				scale += digamma.get(i, j);
			}
			for (int j = 0; j < num_postags + 1; j++) {
				A.set(i, j, (digamma.get(i, j) + smoothing_eps) / scale);
				digamma.set(i, j, 0);
			}

			scale = smoothing_eps * num_words;

			for (int j = 0; j < num_words; j++) {
				scale += gamma_w.get(i, j);
			}
			for (int j = 0; j < num_words; j++) {
				B.set(i, j, (gamma_w.get(i, j) + smoothing_eps) / scale);
				gamma_w.set(i, j, 0);
			}
		}
		normalization(gamma_0);
	}

	/**
	 * Forward algorithm for one sentence s: the sentence alpha: forward
	 * probability matrix of shape (num_postags, max_sentence_length)
	 * 
	 * return: log P(O|\lambda)
	 */
	private double forward(Sentence s) {
		double pprob = 0;
		int T = s.length();
		double[][] fwd = new double[num_postags][T];
		// initialization
		for (int i = 0; i < num_postags; i++) {
			fwd[i][0] = pi.get(0, i)
					* B.get(i, vocabulary.get(s.getWordAt(0).getLemme()));
		}

		for (int t = 0; t <= T - 2; t++) {
			for (int j = 0; j < num_postags; j++) {
				fwd[j][t + 1] = 0;
				for (int i = 0; i < num_postags; i++)
					fwd[j][t + 1] += (fwd[i][t] * A.get(i, j));
				fwd[j][t + 1] *= B.get(j,
						vocabulary.get(s.getWordAt(t + 1).getLemme()));
			}
		}
		alpha = new Matrix(fwd);
		// end
		for (int k = 0; k < num_postags; k++) {
			// System.out.println(fwd[k][T - 1]);
			pprob += fwd[k][T - 1];
		}
		return Math.log(pprob);
	}

	private double forwardScale(Sentence s) {
		for (int i = 0; i < s.length(); i++) {
			int lemme_id = vocabulary.get(s.getWordAt(i).getLemme()).intValue();
			for (int posTag_id = 0; posTag_id < num_postags; posTag_id++) {
				if (i == 0) {
					alpha.set(posTag_id, i,
							pi.get(0, posTag_id) * B.get(posTag_id, lemme_id));
				} else {
					double num = 0;
					for (int prePostag_id = 0; prePostag_id < num_postags; prePostag_id++) {
						num += alpha.get(prePostag_id, i - 1)
								* A.get(prePostag_id, posTag_id)
								* B.get(posTag_id, lemme_id);
					}
					alpha.set(posTag_id, i, num);
				}
			}
			double scale = 0;
			for (int tag_id = 0; tag_id < num_postags; tag_id++) {
				scale += alpha.get(tag_id, i);
			}
			if (scale == 0)
				System.out.println(s.getWordAt(i).getLemme());
			scales.set(0, i, 1 / scale);
			for (int tag_id = 0; tag_id < num_postags; tag_id++) {
				alpha.set(tag_id, i, alpha.get(tag_id, i) * scales.get(0, i));
			}
		}

		double res = 0;
		for (int i = 0; i < s.length(); i++)
			res += Math.log(1 / scales.get(0, i));
		return res;
	}


	/**
	 * Backward algorithm for one sentence
	 * 
	 * return: log P(O|\lambda)
	 */
	private double backward(Sentence s) {
		double pprob = 0;
		int T = s.length();
		double[][] bwd = new double[num_postags][T];

		// initialization
		for (int i = 0; i < num_postags; i++)
			bwd[i][T - 1] = 1;

		for (int t = T - 2; t >= 0; t--) {
			for (int i = 0; i < num_postags; i++) {
				bwd[i][t] = 0;
				for (int j = 0; j < num_postags; j++)
					bwd[i][t] += (bwd[j][t + 1] * A.get(i, j) * B.get(i,
							vocabulary.get(s.getWordAt(t + 1).getLemme())));
			}
		}
		beta = new Matrix(bwd);
		// end
		for (int i = 0; i < num_postags; i++) {
			// System.out.println(bwd[i][0]);
			// System.out.println(pi[i] * B.get(i,
			// vocabulary.get(s.getWordAt(0).getLemme())) * bwd[i][0]);
			pprob += pi.get(0, i)
					* B.get(i, vocabulary.get(s.getWordAt(0).getLemme()))
					* bwd[i][0];
		}
		return Math.log(pprob);
	}

	private double backwardScale(Sentence s) {
		int T = s.length();
		double[][] arr = new double[num_postags][T];
		beta = new Matrix(arr);
		double pprob = 0.0;

		for (int i = 0; i < num_postags; i++) {
			beta.set(i, T - 1, 1.0 / num_postags);
		}

		for (int t = T - 2; t >= 0; t--) {
			for (int j = 0; j < num_postags; j++) {
				double sum = 0.0;
				for (int i = 0; i < num_postags; i++) {
					sum += beta.get(i, t + 1) * A.get(i, j);
				}
				beta.set(
						j,
						t,
						sum
								* B.get(j, vocabulary.get(s.getWordAt(t)
										.getLemme())));
				scales.set(0, t, scales.get(0, t) + beta.get(j, t));
			}
			for (int i = 0; i < num_postags; i++) {
				beta.set(i, t, beta.get(i, t) / scales.get(0, t));
			}
		}

		for (int i = 0; i < num_postags; i++) {
			pprob += beta.get(i, 0);
		}
		return pprob;
	}

	/**
	 * Viterbi algorithm for one sentence v are in log scale, A, B and pi are in
	 * the usual scale.
	 */
	private double viterbi(Sentence s) {
		for (int i = 0; i < s.length(); i++) {
			int lemme_id = vocabulary.get(s.getWordAt(i).getLemme()).intValue();
			for (int tag_id = 0; tag_id < num_postags; tag_id++) {
				if (i == 0) {
					v.set(tag_id,
							i,
							Math.log(pi.get(0, tag_id))
									+ Math.log(B.get(tag_id, lemme_id)));
				} else {
					double Max = Double.NEGATIVE_INFINITY;
					for (int pre_tag_id = 0; pre_tag_id < num_postags; pre_tag_id++) {
						double ele = v.get(pre_tag_id, i - 1)
								+ Math.log(A.get(pre_tag_id, tag_id))
								+ Math.log(B.get(tag_id, lemme_id));
						if (ele > Max) {
							Max = ele;
							back_pointer.set(tag_id, i, pre_tag_id);
						}
					}
					v.set(tag_id, i, Max);
				}
			}
		}

		int res = 0;
		double Max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < num_postags; i++) {
			if (v.get(i, s.length() - 1) > Max) {
				res = i;
				Max = v.get(i, s.length() - 1);
			}
		}

		return res;
	}

	// computes gamma(i, t)
	public double gamma(int i, int t, Sentence s, Matrix fwd, Matrix bwd) {
		double num = fwd.get(i, t) * bwd.get(i, t);
		double denom = 0;

		for (int j = 0; j < num_postags; j++)
			denom += fwd.get(j, t) * bwd.get(j, t);

		return num / denom;
	}

	/**
	 * calculation of probability P(X_t = s_i, X_t+1 = s_j | O, m).
	 * 
	 * @param t
	 *            time t
	 * @param i
	 *            the number of state s_i
	 * @param j
	 *            the number of state s_j
	 * @param o
	 *            an output sequence o
	 * @param fwd
	 *            the Forward-Variables for o
	 * @param bwd
	 *            the Backward-Variables for o
	 * @return PP
	 */
	private double prob(int t, int i, int j, Sentence s, Matrix fwd, Matrix bwd) {
		double num;
		if (t == s.length() - 1)
			num = fwd.get(i, t) * A.get(i, j);
		else
			num = fwd.get(i, t) * A.get(i, j)
					* B.get(j, vocabulary.get(s.getWordAt(t + 1).getLemme()))
					* bwd.get(j, t + 1);

		double denom = 0;

		for (int k = 0; k < num_postags; k++)
			denom += (fwd.get(k, t) * bwd.get(k, t));

		return num / denom;
	}

	public static void main(String[] args) throws IOException {
		// if (args.length < 3) {
		// System.out.println("Expecting at least 3 parameters");
		// System.exit(0);
		// }

		String logFileName = "results/p2/log.txt";
		String labeledFileName = "data/p1/train.txt";
		String unlabeledFileName = "data/p2/unlabeled_20news.txt";
		String predictionFileName = "results/p2/prediction.txt";

		String trainingLogFileName = "results/p2/biglog";

		if (args.length > 3) {
			trainingLogFileName = args[3];
		}

		// read in labeled corpus
		FileHandler fh = new FileHandler();

		ArrayList<Sentence> labeled_corpus = fh
				.readTaggedSentences(labeledFileName);

		ArrayList<Sentence> unlabeled_corpus = fh
				.readUntaggedSentences(unlabeledFileName);

		HMM model = new HMM(labeled_corpus, unlabeled_corpus);
		for (int i = 0; i < 11; i++) {
			model.setMu(0.1*i);
			model.prepareMatrices();
			model.mle();
			model.em();
//			System.out.println(model.mu+"   "+model.predict());
//			model.outputPredictions(predictionFileName + "_"
//					+ String.format("%.1f", model.mu) + ".txt");
			if (trainingLogFileName != null) {
				model.outputTrainingLog(trainingLogFileName + "_"
						+ String.format("%.1f", model.mu) + ".txt");
			}
			
		}
	}

}
