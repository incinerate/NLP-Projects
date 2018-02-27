/**
 * Parser based on the CYK algorithm.
 */

import java.io.*;
import java.security.AllPermission;
import java.util.*;

public class Parser {

	public Cell[][] cell;

	public Grammar g;

	public ArrayList<ArrayList<String>> allSentences = new ArrayList<ArrayList<String>>();
	
	class Cell {

		public String word;
		public ArrayList<Double> prob;
		public ArrayList<String> lhs;
		public ArrayList<RHS> rhs;
		public ArrayList<Cell> leftLeaf;
		public ArrayList<Cell> rightLeaf;
		public boolean isPreterminal;

		public ArrayList<RHS> getRhs() {
			return rhs;
		}
		
		public void setRhs(ArrayList<RHS> rhs) {
			this.rhs = rhs;
		}
		public String getWord() {
			return word;
		}

		public void setWord(String word) {
			this.word = word;
		}

		public ArrayList<Double> getProb() {
			return prob;
		}

		public void setProb(ArrayList<Double> prob) {
			this.prob = prob;
		}

		public ArrayList<String> getLhs() {
			return lhs;
		}

		public void setLhs(ArrayList<String> lhs) {
			this.lhs = lhs;
		}

		public ArrayList<Cell> getLeftLeaf() {
			return leftLeaf;
		}

		public void setLeftLeaf(ArrayList<Cell> leftLeaf) {
			this.leftLeaf = leftLeaf;
		}

		public ArrayList<Cell> getRightLeaf() {
			return rightLeaf;
		}

		public void setRightLeaf(ArrayList<Cell> rightLeaf) {
			this.rightLeaf = rightLeaf;
		}

		public boolean isPreterminal() {
			return isPreterminal;
		}

		public void setPreterminal(boolean isPreterminal) {
			this.isPreterminal = isPreterminal;
		}

		public Cell() {
			isPreterminal = false;
			word = null;
			prob = new ArrayList<Double>();
			lhs = new ArrayList<String>();
			rhs = new ArrayList<RHS>();
			leftLeaf = new ArrayList<>();
			rightLeaf = new ArrayList<>();
		}

		public Integer getId (String s) {
			for (int i = 0; i < lhs.size(); i++)
				if (lhs.get(i).equals(s))
					return i;
			return null;
		}
	}
	/**
	 * Constructor: read the grammar.
	 */
	public Parser(String grammar_filename) {
		g = new Grammar(grammar_filename);
	}

	/**
	 * Parse one sentence given in the array.
	 */
	public void parse(ArrayList<String> sentence) {
		cell = new Cell[sentence.size()][sentence.size()];
		for (int i = 0; i < sentence.size(); i++) {
			for (int j = 0; j < sentence.size() - i; j++) {
				cell[i][j] = new Cell();
				if (i == 0) {
					cell[0][j].isPreterminal = true;
					cell[0][j].word = sentence.get(j);
					ArrayList<String> preTerminals = g
							.findPreTerminals(cell[0][j].word);
					if (preTerminals.size() == 1) {
						cell[0][j].lhs.add(preTerminals.get(0));
						cell[0][j].prob.add(1.0);
					} else {
						for (String str : preTerminals) {
							double totalProb = 0;
							double prob = 0;
							ArrayList<RHS> productions = g.findProductions(str);
							for (RHS rhs : productions) {
								if (rhs.first().equals(cell[0][j].word))
									prob = rhs.getProb();
								totalProb += rhs.getProb();
							}
							cell[i][j].lhs.add(str);
							cell[i][j].prob.add(prob / totalProb);
						}
					}
				} else if (i == 1) {
					for (String lStr1 : cell[i - 1][j].lhs) {
						for (String lStr2 : cell[i - 1][j + 1].lhs) {
							String rhsStr = lStr1 + " " + lStr2;
							if (g.findLHS(rhsStr) != null) {
								for (String str : g.findLHS(rhsStr)) {
									double totalProb = 0;
									double prob = 0;
									ArrayList<RHS> productions = g
											.findProductions(str);
									for (RHS rhs : productions) {
										if (rhs.first().equals(lStr1)) {
											prob = rhs.getProb();
											cell[i][j].rhs.add(rhs);
										}
										totalProb += rhs.getProb();
									}
									cell[i][j].prob.add(prob / totalProb);
								}
								cell[i][j].lhs.addAll(g.findLHS(rhsStr));
								cell[i][j].leftLeaf.add(cell[i - 1][j]);
								cell[i][j].rightLeaf.add(cell[i - 1][j + 1]);
							}
						}
					}
				} else {
					ArrayList<String> arrayList = new ArrayList<String>();
					for (int k = i; k > 0; k--) {
						if (cell[k - 1][j].lhs.size() != 0
								&& cell[i - k][k + j].lhs.size() != 0) {
							for (String lStr1 : cell[k - 1][j].lhs) {
								for (String lStr2 : cell[i - k][k + j].lhs) {
									String rhsStr = lStr1 + " " + lStr2;
									if (g.findLHS(rhsStr) != null) {
										for (String str : g.findLHS(rhsStr)) {
											double totalProb = 0;
											double prob = 0;
											ArrayList<RHS> productions = g
													.findProductions(str);
											for (RHS rhs : productions) {
												if (rhs.first().equals(lStr1)) {
													prob = rhs.getProb();
													cell[i][j].rhs.add(rhs);
												}
												totalProb += rhs.getProb();
											}
											cell[i][j].prob.add(prob
													/ totalProb);
										}
										cell[i][j].lhs.addAll(g.findLHS(rhsStr));
										cell[i][j].leftLeaf.add(cell[k - 1][j]);
										cell[i][j].rightLeaf
										.add(cell[i - k][k + j]);
									}
								}
							}
						}
					}
				}
			}
		}
	}

	/**
	 * Print the parse obtained after calling parse()
	 */
	public String PrintOneParse() {
		return treeGenerator(cell[cell.length - 1][0], "S");
	}

	private String treeGenerator(Cell cell, String s) {
		if (!cell.lhs.contains(s))
			return "ERROR";
		int id = cell.getId(s);
		if (cell.isPreterminal)
			return "(" + s + " " + cell.word + ")";
		if (cell.getLeftLeaf() == null || cell.getRightLeaf() == null) {
			return "ERROR";
		}
		if(cell.getRhs()==null)
			return "ERROR";
		for (RHS rhs : cell.getRhs()) {
			String first = rhs.first();
			String second = rhs.second();
			
			String left = treeGenerator(cell.leftLeaf.get(id), first);
			String right = treeGenerator(cell.rightLeaf.get(id), second);
			if(left.equals("ERROR") || right.equals("ERROR"))
				return "ERROR";
			return "(" + s + " " + left + " " + right + ")";
		}
		return null;
	}

	public static void main(String[] args) throws IOException {
		// read the grammar in the file args[0]
		Parser parser = new Parser("data/grammar.gr");
		List<Character> endList = new ArrayList<Character>();
		FileWriter fw =new FileWriter(new File("results/parse.txt"));
		BufferedWriter bw = new BufferedWriter(fw);

		// read a parse tree from a bash pipe
		try {
			// InputStreamReader isReader = new InputStreamReader(System.in);
			InputStreamReader isReader = new InputStreamReader(
					new FileInputStream("results/sentence.txt"));
			BufferedReader bufReader = new BufferedReader(isReader);
			while (true) {
				ArrayList<String> sentence = new ArrayList<String>();
				String line = null;
				if ((line = bufReader.readLine()) != null) {
					char end =line.charAt(line.length()-2);
					endList.add(end);
					String[] words = line.split(" ");
					for (String word : words) {
						word = word.replaceAll("[^a-zA-Z]", "");
						if (word.length() == 0) {
							continue;
						}
						// use the grammar to filter out non-terminals and
						// pre-terminals
						if (parser.g.symbolType(word) == 0
								&& (!word.equals(".") && !word.equals("!"))) {
							sentence.add(word);
						}
					}
					parser.allSentences.add(sentence);
				} else {
					break;
				}
			}
			bufReader.close();
			isReader.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		int i=0;
		for (ArrayList<String> s : parser.allSentences) {
			parser.parse(s);
			Character end = endList.get(i);
			i++;
			System.out.println("(ROOT " + parser.PrintOneParse() + " " + end + ")");
			bw.write("(ROOT " + parser.PrintOneParse() + " " + end + ")\n");
		}
		bw.flush();
		bw.close();
	}
}
