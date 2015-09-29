package keel.Algorithms.Discretizers.ecpsd;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

/**
 * <p>Title: ECPSD </p>
 *
 * <p>Description: It contains the implementation of the CHC multivariate discretizer with a historical of cut points.
 * It makes a faster convergence of the algorithm CHC_MV. </p>
 * 
 * <p>Company: KEEL </p>
 *
 * @author Written by Sergio Ramirez (University of Granada) (10/01/2014)
 * @version 1.5
 * @since JDK1.5
 */

public class EMD implements Serializable{
	
	/**
	 * 
	 */
	private long seed;
	private Chromosome initial_chr;
	private Chromosome best;
	private float[][] cut_points;
	
	public ArrayList <Chromosome> population;
	public ArrayList <Chromosome> pop_to_eval;
	
	//private int max_cut_points;
	private int n_cut_points;
	
	private int pop_length;
	private int nClasses;
	private weka.core.Instances baseTrain;
	
	private float r;
	private float alpha;
	private float best_fitness;
	private float prob1to0Rec;
	private float prob1to0Div;  
	
	/* Execution parameters */
	private float threshold;
	public int n_eval;
	private int n_restart_not_improving;
	private int max_eval;
    private boolean needs_eval = false;
    /**
     * Creates a CHC object with its parameters
     * 
     * @param pop	Population of rules we want to select
     */
    public EMD (long seed, float[][] cut_points, int eval, int popLength, float restart_per, 
    		float alpha_fitness, float beta_fitness, float pr0to1Rec, 
    		float pr0to1Div, int nClasses, boolean[] initial_chr) {
    	
    	this.seed = seed;
    	//this.original_cut_points = cut_points.clone();
    	max_eval = eval;
    	pop_length = popLength;
    	r = restart_per;
    	alpha = alpha_fitness;
    	prob1to0Rec = pr0to1Rec;
    	prob1to0Div = pr0to1Div;
    	this.nClasses = nClasses;
    	this.cut_points = cut_points; 
    	
    	n_cut_points = 0;
    	for (int i=0; i< cut_points.length; i++) {
    		if (cut_points[i] != null) {
    			//if(!isAscendingSorted(cut_points[i]))
    			//		throw new ExceptionInInitializerError("Cut points must be sorted");
    			n_cut_points += cut_points[i].length;
    		}
    			
    	}
    	
    	population = new ArrayList <Chromosome> (pop_length);
    	best_fitness = 100f;
    	if(initial_chr == null) {
    		this.initial_chr = new Chromosome (n_cut_points, true);
    	} else {
    		if(initial_chr.length == n_cut_points)
        		this.initial_chr = new Chromosome (initial_chr);
        	else 
        		this.initial_chr = new Chromosome (n_cut_points, true);
    	}
    	pop_to_eval = population;
    	
    	this.baseTrain = computeBaseTrain();
    	this.n_eval = 0;
    	this.threshold = (float) n_cut_points / 4.f;
    	this.n_restart_not_improving = 0;
    	
    }
    
    public EMD (float[][] cut_points, int chLength, int nEval, int nClasses) {    	
    	this(964534618L, cut_points, nEval, 50, .8f, .7f, .3f, .25f, .05f, nClasses, null);
    }
    
    public EMD (float[][] cut_points, boolean[] initial_chr, float alpha, int nEval, int nClasses) {
    	this(964534618L, cut_points, nEval, 50, .8f, alpha, 1-alpha, .25f, .05f, nClasses, initial_chr);
    }
    
    private weka.core.Instances computeBaseTrain() {
    	/* WEKA data set initialization
    	 * Second and Third type of evaluator in precision: WEKA classifier
    	 * */    	
	    ArrayList<weka.core.Attribute> attributes = new ArrayList<weka.core.Attribute>();
	    //double[][] ranges = dataset.getRanges();
	    //weka.core.Instance instances[] = new weka.core.Instance[discretized_data.length];
	    
	    /*Attribute adaptation to WEKA format*/
	    for (int i=0; i< cut_points.length; i++) {
	    	List<String> att_values = new ArrayList<String>();
    		if(cut_points[i] != null) {
	    		for (int j=0; j < cut_points[i].length + 1; j++)
		    		att_values.add(new String(Integer.toString(j)));
    		} else {
    			//for (int j = (int) ranges[i][0]; j <= ranges[i][1]; j++) 
    				//att_values.add(new String(Integer.toString(j)));
    			att_values.add("0");
    		}
    		weka.core.Attribute att = 
	    			new weka.core.Attribute("At" + i, att_values, i);
    	    attributes.add(att);
	    }
	    
	    List<String> att_values = new ArrayList<String>();
	    for (int i=0; i<nClasses; i++) {
	    	att_values.add(new String(Integer.toString(i)));
	    }
	    attributes.add(new weka.core.Attribute("Class", att_values, cut_points.length));

    	/*WEKA data set construction*/
	    weka.core.Instances baseTrain = new weka.core.Instances(
	    		"CHC_evaluation", attributes, 0);
	    baseTrain.setClassIndex(attributes.size() - 1);
    	return baseTrain;
    }
    
    /**
     * Run the CHC algorithm for the data in this population
     * 
     * @return	boolean array with the rules selected for the final population
     */
    /*public void runAlgorithm() {
    	ArrayList <Chromosome> C_population;
    	ArrayList <Chromosome> Cr_population;
    	boolean pop_changes;
    	
    	initPopulation();
		evalPopulation();
		
    	do {
    		
    		// Select for crossover
    		C_population = randomSelection();
    		// Cross selected individuals
    		Cr_population = recombine (C_population);
    		// Evaluate new population
    		 evaluate (Cr_population);
    		
    		// Select individuals for new population
    		pop_changes = selectNewPopulation (Cr_population);
    		
    		// Check if we have improved or not
    		if (!pop_changes) threshold--;
    		
    		// If we do not improve our current population for several trials, then we should restart the population
    		if (threshold < 0) {
    			restartPopulation();
    			threshold = Math.round(r * (1.0 - r) * (float) n_cut_points);
    	    	best_fitness = 100.f;
    			n_restart_not_improving++;
    			evalPopulation();
    		}
    	} while ((n_eval < max_eval) && (n_restart_not_improving < 5));

    	// The evaluations have finished now, so we select the individual with best fitness
    	Collections.sort(population);
    	Chromosome best = population.get(0);
    	
    	this.best = best;
    }*/

    
    /**
     * Creates several population individuals randomly. The first individual has all its values set to true
     */
    public void initPopulation () {    	
    	population.add(initial_chr);    	
    	for (int i=1; i<pop_length; i++)
    		population.add(new Chromosome(n_cut_points));
    	needs_eval = true; pop_to_eval = population;
    }
    
    public ArrayList<Chromosome> crossover() {    	
    	// Select for crossover
    	ArrayList<Chromosome> C_population = randomSelection();
		// Cross selected individuals
    	ArrayList<Chromosome> Cr_population = recombine (C_population);
		needs_eval = true; pop_to_eval = Cr_population;
		return Cr_population;
    }
    
    public boolean newPopulation(ArrayList<Chromosome> newPop) {
    	// Select individuals for new population
		boolean pop_changes = selectNewPopulation (newPop);
		// Check if we have improved or not
		if (!pop_changes) threshold--;
		
		// If we do not improve our current population for several trials, then we should restart the population
		if (threshold < 0) {
			restartPopulation();
			threshold = Math.round(r * (1.0 - r) * (float) n_cut_points);
	    	best_fitness = 100.f;
			n_restart_not_improving++;
			needs_eval = true; pop_to_eval = population;
			return true; // Need an extra evaluation!
		}
		return false;
    }
    
    public boolean isFinished() {
    	return (n_eval >= max_eval) || (n_restart_not_improving >= 5);
    }
    
    public boolean needsToEval(){
    	return needs_eval;
    }
    
    public ArrayList<Chromosome> getPopulation() {
		return population;
	}
    
    public Chromosome getBest(){
    	Collections.sort(population);
    	return population.get(0);
    }
    
    /**
     * Evaluates the population individuals. If a chromosome was previously evaluated we do not evaluate it again (read-only method)
     */
    public EvalPoint[] evalPopulation (float[][] dataset) {
    	EvalPoint[] result = new EvalPoint[pop_to_eval.size()];
    	if(needs_eval) {        	
            for (int i = 0; i < pop_to_eval.size(); i++) {
                if (pop_to_eval.get(i).not_eval()) {
                	result[i] = pop_to_eval.get(i).evaluate(this.baseTrain, dataset, cut_points);
                	//n_eval++;
                }
            }
    	}
        return result;
    }
    
    public void setFitness(EvalPoint[] fitnesses) {    	
    	for (int i = 0; i < pop_to_eval.size(); i++) {
    		if (pop_to_eval.get(i).not_eval()) {    			
    			boolean[] ind = pop_to_eval.get(i).getIndividual();
        		int nsel = 0;
        		for (int j = 0; j < ind.length; j++) {
    				if(ind[j]) nsel++;
    			}
        		pop_to_eval.get(i).setFitness(fitnesses[i], nsel, n_cut_points, alpha);
        		float ind_fitness = pop_to_eval.get(i).getFitness();
            	if (ind_fitness < best_fitness) {
            		best_fitness = ind_fitness;            		
            	}
            	n_eval++;
    		}    		
    	}
    	needs_eval = false;
    }
    
    /**
     * Selects all the members of the current population to a new population ArrayList in random order
     * 
     * @return	the current population in random order
     */
    private ArrayList <Chromosome> randomSelection() {
    	ArrayList <Chromosome> C_population;
    	int [] order;
    	int pos, tmp;
    	
    	C_population = new ArrayList <Chromosome> (pop_length);
    	order = new int[pop_length];
    	
    	for (int i=0; i<pop_length; i++) {
    		order[i] = i;
    	}
    	
    	Random randomGenerator = new Random(seed);
    	for (int i=0; i<pop_length; i++) {
    		int max = pop_length;
    		int min = i;
    		pos = randomGenerator.nextInt(max - min) + min;
    		tmp = order[i];
    		order[i] = order[pos];
    		order[pos] = tmp;
    	}
    	
    	for (int i=0; i<pop_length; i++) {
    		C_population.add(new Chromosome(((Chromosome)population.get(order[i]))));
    	}
    	
    	return C_population;
    }
    
    /**
     * Obtains the descendants of the given population by creating the most different descendant from parents which are different enough
     * 
     * @param original_population	Original parents used to create the descendants population
     * @return	Population of descendants of the given population
     */
    private ArrayList <Chromosome> recombine (ArrayList <Chromosome> original_population) {
    	ArrayList <Chromosome> Cr_population;
    	int distHamming, n_descendants;
    	Chromosome main_parent, second_parent;
    	ArrayList <Chromosome> descendants;
    	
    	n_descendants = pop_length;
    	if ((n_descendants%2)!=0)
    		n_descendants--;
    	Cr_population = new ArrayList <Chromosome> (n_descendants);
    	
    	for (int i=0; i<n_descendants; i+=2) {
    		main_parent = (Chromosome)original_population.get(i);
    		second_parent = (Chromosome)original_population.get(i+1);
    		
    		distHamming = main_parent.hammingDistance(second_parent);
    		
    		if ((distHamming/2.0) > threshold) {
    			descendants = main_parent.createDescendants(second_parent, prob1to0Rec);
    			//descendants = main_parent.createDescendants(second_parent);
    			Cr_population.add((Chromosome)descendants.get(0));
    			Cr_population.add((Chromosome)descendants.get(1));
    		}
    	}
    	
    	return Cr_population;
    }
    
    /**
     * Evaluates the given individuals. If a chromosome was previously evaluated we do not evaluate it again
     * 
     * @param pop	Population of individuals we want to evaluate
     */
    /*private void evaluate (ArrayList <Chromosome> pop) {
    	for (int i = 0; i < pop.size(); i++) {
            if (pop.get(i).not_eval()) {
            	pop.get(i).evaluate(baseTrain, dataset, cut_points, max_cut_points, alpha, beta);
            	n_eval++;
            }
        }
    }*/
    
    /**
     * Replaces the current population with the best individuals of the given population and the current population
     * 
     * @param pop	Population of new individuals we want to introduce in the current population
     * @return true, if any element of the current population is changed with other element of the new population; false, otherwise
     */
    private boolean selectNewPopulation (ArrayList <Chromosome> pop) {
    	float worst_old_population, best_new_population;
    	
    	// First, we sort the old and the new population
    	Collections.sort(population);
    	Collections.sort(pop);
    	
    	worst_old_population = ((Chromosome)population.get(population.size()-1)).getFitness();
    	if (pop.size() > 0) {
    		best_new_population = ((Chromosome)pop.get(0)).getFitness();
    	}
    	else {
    		best_new_population = 0.f;
    	}	
    	
    	//if ((worst_old_population >= best_new_population) || (pop.size() <= 0)) {
    	if ((worst_old_population <= best_new_population) || (pop.size() <= 0)) {
    		return false;
    	} else {
    		ArrayList <Chromosome> new_pop;
    		Chromosome current_chromosome;
    		int i = 0;
    		int i_pop = 0;
    		boolean copy_old_population = true;
    		boolean small_new_pop = false;
    		
    		new_pop = new ArrayList <Chromosome> (pop_length);
    		
    		// Copy the members of the old population better than the members of the new population
    		do {
    			current_chromosome = (Chromosome)population.get(i);
    			float current_fitness = current_chromosome.getFitness();
    			
    			//if (current_fitness < best_new_population) {
    			if (current_fitness >= best_new_population) {
    				// Check if we have enough members in the new population to create the final population
    				if ((pop_length - i) > pop.size()) {
    					new_pop.add(current_chromosome);
        				i++;
        				small_new_pop = true;
    				} else {
    					copy_old_population = false;
    				}
    			} else {
    				new_pop.add(current_chromosome);
    				i++;
    			}
    		} while ((i < pop_length) && (copy_old_population));
    		
    		while (i < pop_length) {
    			current_chromosome = (Chromosome)pop.get(i_pop);
    			new_pop.add(current_chromosome);
    			i++;
    			i_pop++;
    		}
    		
    		if (small_new_pop) {
    			Collections.sort(new_pop);
    		}
    		
    		float current_fitness = ((Chromosome)new_pop.get(0)).getFitness();
    		
    		if (best_fitness > current_fitness) {
    			best_fitness = current_fitness;
    			n_restart_not_improving = 0;
    		}
    		
    		population = new_pop;	
        	return true;
    	}
    }
    
    /**
     * Creates a new population using the CHC diverge procedure
     */
    private void restartPopulation () {
    	ArrayList <Chromosome> new_pop;
    	Chromosome current_chromosome;
    	
    	new_pop = new ArrayList <Chromosome> (pop_length);
    	
    	Collections.sort(population);
    	current_chromosome = (Chromosome)population.get(0);
    	new_pop.add(current_chromosome);
    	
    	for (int i=1; i<pop_length; i++) {
    		//current_chromosome = new CHC_Chromosome (
    		//		(CHC_Chromosome)population.get(0), r);
    		current_chromosome = new Chromosome (
    				(Chromosome)population.get(0), r, prob1to0Div);
    		new_pop.add(current_chromosome);
    	}
    	
    	population = new_pop;
    }
    
    class ComparePointsByID implements Comparator<RankingPoint> {
        @Override
        public int compare(RankingPoint o1, RankingPoint o2) {
        	return new Integer(o1.id).compareTo(o2.id);
        }
    }

    // Ascending
    class ComparePointsByRank implements Comparator<RankingPoint> {
        @Override
        public int compare(RankingPoint o1, RankingPoint o2) {
        	return new Integer(o2.rank).compareTo(o1.rank);
        }
    }
}

