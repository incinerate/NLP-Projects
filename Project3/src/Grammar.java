/**
 * @author sihong
 * A helper class to read CNF grammars.
 * 
 * Use a hash table to store a grammar:
 * 1) the keys are the LHS
 * 2) the value for a key is a list of lists of strings. Each string represents a terminal or non-terminal.
 * 		multiple RHSes for a single LHS are allowed.
 */

import java.io.*;
import java.util.*;

public class Grammar {
	
	private Hashtable<String, ArrayList< RHS> > grammar;
	
	private HashSet<String> terminals = null;
	
	private HashSet<String> preTerminals = null;
	
	private HashSet<String> nonTerminals = null;
	
	/* Record all POS tags for each word: a word can have multiple POS tags */
	private Hashtable<String, ArrayList<String>> word2tag = null;
	
	/* Record all LHS for valid RHS:
	 * 	the same rhs can be derived from multiple different LHSes.
	 * 	a rhs here is represented by a string of the symbols in RHS concatenated by a space (e.g., VP NP).
	 * */
	private Hashtable<String, ArrayList<String>> rhs2lhs = null;

	public Grammar(String filename) {
		grammar = new Hashtable<String, ArrayList<RHS > >();
		nonTerminals = new HashSet<String>();
		preTerminals = new HashSet<String>();
		terminals = new HashSet<String>();
		
		word2tag = new Hashtable<String, ArrayList<String> >();
		
		rhs2lhs = new Hashtable<String, ArrayList<String>>();
		
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(filename));
			String line;
			while ((line = br.readLine()) != null) {
				// ignore comments and empty lines
				if (line.trim().length() == 0 || line.charAt(0) == '#') {
					continue;
				}
				
				String[] entries = line.split("\\s+");

				double count = Double.parseDouble(entries[0]);
				
				for (int i = 1; i < entries.length; ++i) {
					int type = symbolType(entries[i]);
					switch (type) {
						case 0:
							terminals.add(entries[i]);
							if (!word2tag.containsKey(entries[i])) {
								word2tag.put(entries[i], new ArrayList<String>());
							}
							// assume that entries[1] is the LHS and entries[i] is on the RHL
							word2tag.get(entries[i]).add(entries[1]);
							break;
						case 1:
							preTerminals.add(entries[i]);
							break;
						case 2:
							nonTerminals.add(entries[i]);
							break;
					}
				}
				
				String lhs = entries[1];
				
				// a RHS is a list of terminals/non-terminals.
				ArrayList<String> symbols = new ArrayList<String>();
				for (int i = 2; i < entries.length; ++i) {
					symbols.add(entries[i]);
				}
				
				RHS rhs = new RHS(symbols, count);
				
				// add production (lhs -> rhs) to grammar
				if (!grammar.containsKey(lhs)) {
					grammar.put(lhs, new ArrayList<RHS > ());
				}
				grammar.get(lhs).add(rhs);

				// add reverse production (rhs -> lhs)
				String rhsStr = rhs.first() + " " + rhs.second();
				if (!rhs2lhs.containsKey(rhsStr)) {
					rhs2lhs.put(rhsStr, new ArrayList<String>());
				}
				rhs2lhs.get(rhsStr).add(lhs);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		// normalize the counts to prob
		for (String lhs : grammar.keySet()) {
			double sum = 0;
			for (RHS rhs : grammar.get(lhs)) {
				sum += rhs.getProb();
			}
			if (sum == 0) {
				sum = 1;
			}
			for (RHS rhs : grammar.get(lhs)) {
				rhs.setProb(rhs.getProb() / sum);
			}
		}
	}
	
	/**
	 * Check the type of a symbol
	 * @param symbol
	 * @return
	 *  0 if terminal
	 *  1 if pre-terminal
	 *  2 if non-terminal
	 */
	public int symbolType(String symbol) {
		if (terminals != null && terminals.contains(symbol)) {
			return 0;
		} else if (nonTerminals != null && nonTerminals.contains(symbol)) {
			return 2;
		} else if (preTerminals != null && preTerminals.contains(symbol)) {
			return 1;
		}
		
		if (Character.isUpperCase(symbol.charAt(0))) {
			if (symbol.length() == 1 || Character.isUpperCase(symbol.charAt(1)) ) {
				// nonTerminal: the first two chars are cap.
				return 2;
			} else {
				// preTerminal: only the first char is cap.
				return 1;
			}
		} else {
			// terminal: all lower case
			return 0;
		}
	}
	
	/**
	 * Find suitable POS tags for a word. 
	 */
	public ArrayList<String> findPreTerminals(String word) {
		return word2tag.get(word);
	}
	
	/**
	 * Find any LHSes (such as "S" and "VP") given the RHS string (such as "VP NP")
	 * @param rhsStr
	 * @return
	 */
	public ArrayList<String> findLHS(String rhsStr) {
		return rhs2lhs.get(rhsStr);
	}
	
	public void printGrammarInfo() {
		
		System.out.println("Grammar info:\n");
		int num_rules = 0;
		for (String key : grammar.keySet()) {
			num_rules += grammar.get(key).size();
		}
		System.out.println("Grammar loaded:");
		
		System.out.println("Number of rules: " + num_rules);
		System.out.println("Number of terminals: " + terminals.size());
		for (String s : terminals) {
			System.out.println(s);
		}
		System.out.println();
		
		System.out.println("Number of preterminals: " + preTerminals.size());
		for (String s : preTerminals) {
			System.out.println(s);
		}
		System.out.println();
		
		System.out.println("Number of non-terminals: " + nonTerminals.size());
		for (String s : nonTerminals) {
			System.out.println(s);
		}
		System.out.println();
		
		for (String lhs : grammar.keySet()) {
			//System.out.println(s);
			for (RHS rhs : grammar.get(lhs)) {
				rhs.printProduction(lhs);
			}
		}
	}
	
	/**
	 * Retrieve all RHSes of the given LHS.
	 * @param rhs
	 * @return
	 */
	public ArrayList<RHS > findProductions(String lhs) {
		if (grammar.containsKey(lhs)) {
			return grammar.get(lhs);
		} else {
			return null;
		}
	}
	
	/**
	 * Check if a particular combination of two non-terminals is a valid RHS
	 * @param args
	 */
	public static void main(String[] args) {
		Grammar g = new Grammar("data/grammar.gr");
		g.printGrammarInfo();
	}

	public Hashtable<String, ArrayList<RHS>> getGrammar() {
		return grammar;
	}

	public void setGrammar(Hashtable<String, ArrayList<RHS>> grammar) {
		this.grammar = grammar;
	}

	public HashSet<String> getTerminals() {
		return terminals;
	}

	public void setTerminals(HashSet<String> terminals) {
		this.terminals = terminals;
	}

	public HashSet<String> getPreTerminals() {
		return preTerminals;
	}

	public void setPreTerminals(HashSet<String> preTerminals) {
		this.preTerminals = preTerminals;
	}

	public HashSet<String> getNonTerminals() {
		return nonTerminals;
	}

	public void setNonTerminals(HashSet<String> nonTerminals) {
		this.nonTerminals = nonTerminals;
	}

	public Hashtable<String, ArrayList<String>> getWord2tag() {
		return word2tag;
	}

	public void setWord2tag(Hashtable<String, ArrayList<String>> word2tag) {
		this.word2tag = word2tag;
	}

	public Hashtable<String, ArrayList<String>> getRhs2lhs() {
		return rhs2lhs;
	}

	public void setRhs2lhs(Hashtable<String, ArrayList<String>> rhs2lhs) {
		this.rhs2lhs = rhs2lhs;
	}
	
	
}
