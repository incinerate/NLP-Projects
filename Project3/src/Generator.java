/**
 * Generate sentences from a CFG
 * 
 * @author sihong
 *
 */

import java.io.*;
import java.util.*;

public class Generator {

	private Grammar grammar;

	/**
	 * Constructor: read the grammar.
	 */
	public Generator(String grammar_filename) {
		grammar = new Grammar(grammar_filename);
	}

	/**
	 * Generate a number of sentences.
	 */
	public ArrayList<String> generate(int numSentences) {
		ArrayList<String> l = new ArrayList<String>();
		for (int i = 0; i < numSentences; i++) {
			ArrayList<RHS> productions = grammar.findProductions("ROOT");
			l.add("(ROOT ");
			generateSentences(l, productions);
			l.add(System.getProperty("line.separator"));
		}
		return l;
	}

	private void generateSentences(ArrayList<String> l,
			ArrayList<RHS> productions) {
		boolean nounadj = false;
		if (productions == null)
			l.add(")");
		else {
			RHS rhs = createGrammarByProb(productions);
			grammar.findLHS(rhs.first() + " " + rhs.second()).get(0);
			if (grammar.symbolType(grammar.findLHS(
					rhs.first() + " " + rhs.second()).get(0)) == 1
					&& grammar.symbolType(rhs.first()) == 1)
				nounadj = true;
			if (grammar.symbolType(rhs.first()) == 2) {
				l.add("(" + rhs.first() + " ");
				generateSentences(l, grammar.findProductions(rhs.first()));
				l.add(") ");
				if (!(rhs.second().equals("!") || rhs.second().equals(".")))
					l.add("(");
				l.add(rhs.second());
				if (!(rhs.second().equals("!") || rhs.second().equals(".")))
					l.add(" ");
				generateSentences(l, grammar.findProductions(rhs.second()));
				if (!(rhs.second().equals("!") || rhs.second().equals(".")))
					l.add("))");
			} else if (grammar.symbolType(rhs.first()) == 1) {
				l.add("(" + rhs.first() + " ");
				generateSentences(l, grammar.findProductions(rhs.first()));
				l.add(" (" + rhs.second() + " ");
				generateSentences(l, grammar.findProductions(rhs.second()));
				if(nounadj) l.add(")");
			} else {
				l.add(rhs.first() + ")");
			}
		}

	}

	private RHS createGrammarByProb(ArrayList<RHS> productions) {
		RHS rhs = productions.get(0);
		if (productions.size() > 1) {
			double sum = Math.random();
			double prob = 0;
			for (int i = 0; i < productions.size(); i++) {
				if (prob + productions.get(i).getProb() >= sum) {
					rhs = productions.get(i);
					break;
				} else {
					prob += productions.get(i).getProb();
				}
			}
		}
		return rhs;
	}

	public static void main(String[] args) throws Exception {
		FileWriter fw = new FileWriter(new File("results/sentence.txt"));
		BufferedWriter bw = new BufferedWriter(fw);
		// the first argument is the path to the grammar file.
		Generator g = new Generator("data/grammar.gr");
		ArrayList<String> res = g.generate(10);
		for (String s : res) {
			bw.write(s);
		}
		bw.flush();
		bw.close();
	}
}
