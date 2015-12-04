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
 * @author Modified by Sergio Ramirez (University of Granada) (04/12/2015)
 * @version 1.5
 * @since JDK1.5
 */

public class EMD implements Serializable{

	private static final long serialVersionUID = 7575712219028489742L;
	private long seed;
	private float[][] cut_points;
	private float[][] original_cut_points;
	private float[][] dataset;
	private Chromosome initial_chr;
	private Chromosome best;
	
	private ArrayList <Chromosome> population;
	
	private int max_cut_points;
	private int n_cut_points;
	
	private int max_eval;
	private int n_eval;
	
	private int pop_length;
	private int nClasses;
	private weka.core.Instances baseTrain;
	
	private float threshold;
	private float r;
	private float alpha, beta;
	private float best_fitness;

	private float prob1to0Rec;
	private float prob1to0Div;  
	private int n_restart_not_improving;
	
	// Reduction 
	private static float pReduction = .5f;
	private static float pEvaluationsForReduction = .1f;
	private static int PROPER_SIZE_CHROMOSOME = 1000;
    
    /**
     * Creates a CHC object with its parameters
     * 
     * @param seed Seed initialization parameter
     * @param dataset Input data in tabular format
     * @param cut_points Cut points to evaluate in tabular format
     * @param eval Number of evaluations for GA
     * @param popLength Number of chromosomes
     * @param restart_per Restar percentage for population
     * @param alpha_fitness Weight factor for fitness evaluation
     * @param nClasses Number of classes 
     * @param initial_chr Initial chromosome
     * 
     */
    public EMD (long seed, float[][] dataset, float [][] cut_points, int eval, int popLength, 
    		float restart_per, float alpha_fitness, int nClasses, boolean[] initial_chr) {
    	
    	this.seed = seed;
    	this.dataset = dataset;
    	this.cut_points = cut_points;
    	this.original_cut_points = cut_points.clone();
    	max_eval = eval;
    	pop_length = popLength;
    	r = restart_per;
    	alpha = alpha_fitness;
    	this.nClasses = nClasses;
    	
    	max_cut_points = 0;
    	for (int i=0; i< cut_points.length; i++) {
    		if (cut_points[i] != null) {
    			//if(!isAscendingSorted(cut_points[i]))
    			//		throw new ExceptionInInitializerError("Cut points must be sorted");
    			max_cut_points += cut_points[i].length;
    		}
    			
    	}
    	n_cut_points = max_cut_points;
    	
    	population = new ArrayList <Chromosome> (pop_length);
    	best_fitness = 100f;
    	if(initial_chr == null) {
    		this.initial_chr = new Chromosome (n_cut_points, true);
    	} else {
    		if(initial_chr.length == max_cut_points)
        		this.initial_chr = new Chromosome (initial_chr);
        	else 
        		this.initial_chr = new Chromosome (n_cut_points, true);
    	}
    	
    	baseTrain = computeBaseTrain();
    	
    }
    
    public EMD (float[][] current_dataset, float [][] cut_points, int nEval, int nClasses) {    	
    	this(964534618L, current_dataset, cut_points, nEval, 
    			50, .8f, .7f, nClasses, null);
    }
    
    public EMD (float[][] current_dataset, float [][] cut_points, boolean[] initial_chr, float alpha, int nEval, int nClasses) {
    	this(964534618L, current_dataset, cut_points, nEval, 
    			50, .8f, alpha, nClasses, initial_chr);
    }
    
    /**
     * Creates the attributes header for the subsequent evaluations on WEKA
     * @return An attributes header for WEKA.
     */
    private weka.core.Instances computeBaseTrain() {
    	int nInputs = dataset[0].length - 1;
    	/* WEKA data set initialization
    	 * Second and Third type of evaluator in precision: WEKA classifier
    	 * */    	
	    ArrayList<weka.core.Attribute> attributes = new ArrayList<weka.core.Attribute>();
	    //double[][] ranges = dataset.getRanges();
	    //weka.core.Instance instances[] = new weka.core.Instance[discretized_data.length];
	    
	    /*Attribute adaptation to WEKA format*/
	    for (int i=0; i< nInputs; i++) {
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
	    attributes.add(new weka.core.Attribute("Class", att_values, nInputs));

    	/*WEKA data set construction*/
	    weka.core.Instances baseTrain = new weka.core.Instances(
	    		"CHC_evaluation", attributes, 0);
	    baseTrain.setClassIndex(attributes.size() - 1);
    	return baseTrain;
    }
    
    /**
     * Run EMD algorithm on the input data
     */
    public void runAlgorithm() {
    	ArrayList <Chromosome> C_population;
    	ArrayList <Chromosome> Cr_population;
    	boolean pop_changes;
    	
    	n_eval = 0;
    	threshold = (float) n_cut_points / 4.f;
    	n_restart_not_improving = 0;
    	int n_reduction = 0;
    	int n_restart = 0;
    	int next_reduction = 1;
    	boolean reduction = true;

    	int[] cut_points_log = new int[n_cut_points];   	
    	
    	initPopulation();
		evalPopulation();
		
		do {
    		
    		// Reduction?
    		reduction = (n_cut_points * (1 - pReduction) > PROPER_SIZE_CHROMOSOME) &&
    				(n_eval / (max_eval * pEvaluationsForReduction) > next_reduction);
    		if (reduction) {
    			
    			System.out.println("Reduction!");
    			// We reduce the population, and it is not evaluated this time
    			reduction(cut_points_log, ((Chromosome)population.get(0)).getIndividual());    	    	
    			cut_points_log = new int[n_cut_points];
    			next_reduction++;

    			// Next time we do a restart 
    			// (population do not need be evaluated, best chrom is keeped equal)
    			restartPopulation();
    			
    			threshold = Math.round(r * (1.0 - r) * (float) n_cut_points);
    	    	best_fitness = 100.0f;
    	    	
    	    	baseTrain = computeBaseTrain();
    			evalPopulation();    			
    			n_reduction++;
    			if(n_cut_points * (1 - pReduction) <= PROPER_SIZE_CHROMOSOME) {
    				System.out.println("No more reductions!");
    			}
    			Collections.sort(population);  
    		}
    		
    		// Select for crossover
    		C_population = randomSelection();
    		
    		// Cross selected individuals
    		Cr_population = recombine (C_population);
    		// Evaluate new population
    		 evaluate (Cr_population);
    		 
    		// Select individuals for new population
    		pop_changes = selectNewPopulation (Cr_population);
    		
    		// Maintain a historical of the most selected cut points after selecting new populations
    		if (reduction) {
	    		for(int i=0; i < n_cut_points; i++) {
	    			if(population.get(0).getIndividual()[i]) 
	    				cut_points_log[i]++;
	    		}
    		}
    		
    		// Check if we have improved or not
    		if (!pop_changes) threshold--;
    		
    		// If we do not improve our current population for several trials, then we should restart the population
    		if (threshold < 0) {
    			restartPopulation();
    			threshold = Math.round(r * (1.0 - r) * (float) n_cut_points);
    	    	best_fitness = 100.f;
    			n_restart_not_improving++;
    			evalPopulation();
    			n_restart++;
    		}
    		
    	} while ((n_eval < max_eval) && (n_restart_not_improving < 5));

    	// The evaluations have finished now, so we select the individual with best fitness
    	Collections.sort(population);
    	best = population.get(0);
    }
    
    public float getBestError() {
    	return best.perc_err;
    }
    
    public float getBestFitness(){
    	return best.getFitness();
    }
    
    public int getCurrentSize(){
    	return best.getIndividual().length;
    }
    
    /**
     * Return the best individual after being executed the GA algorithm.
     * If there have been a reduction in the chromosome, the process is reverted.
     * 
     * @return The best chromosome and the points selected.
     */
    public boolean[] getBestIndividual(){
    	if(max_cut_points == best.getIndividual().length) 
    		return best.getIndividual();
    	
		boolean[] result = new boolean[max_cut_points];
		boolean[] bred = best.getIndividual();
		for(int i = 0, acc = 0, oacc = 0; i<cut_points.length; i++){
			if(cut_points[i] != null) {
				for(int j = 0, pind = 0; j<cut_points[i].length;j++) {
    				if(bred[acc + j]) { // The point's been chosen
    					float[] oatt = original_cut_points[i];
						// Search the correspondent index
    					boolean found = false;
						while(pind < oatt.length && !found){
    						if(oatt[pind] == cut_points[i][j]) found = true;
    						pind++;
    					}
						if(found) result[oacc + pind - 1] = true;        					
    				}
    			}
				acc += cut_points[i].length;
				oacc += original_cut_points[i].length;
			}
		}
		return result;
    }
    
    /**
     * Creates several population individuals randomly. The first individual has all its values set to true
     */
    private void initPopulation () {
    	population.add(initial_chr);    	
    	for (int i=1; i<pop_length; i++)
    		population.add(new Chromosome(n_cut_points));
    }
    
    /**
     * Evaluates the population individuals. If a chromosome was previously evaluated we do not evaluate it again
     */
    private void evalPopulation () {
    	
        for (int i = 0; i < pop_length; i++) {
            if (population.get(i).not_eval()) {
                //program.execute(args[0]);
            	population.get(i).evaluate(baseTrain, dataset, cut_points, alpha, beta);
            	//population.get(i).evaluate(dataset, cut_points, max_cut_points, alpha, beta);
            	n_eval++;
            }
        	
        	float ind_fitness = population.get(i).getFitness();
        	if (ind_fitness < best_fitness) {
        		best_fitness = ind_fitness;            		
        	}
        }
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
    private void evaluate (ArrayList <Chromosome> pop) {
    	for (int i = 0; i < pop.size(); i++) {
            if (pop.get(i).not_eval()) {
            	pop.get(i).evaluate(baseTrain, dataset, cut_points, alpha, beta);
            	n_eval++;
            }
        }
    }
    
    /**
     * Reduction process applied on the chromosomes. Those points with the best rank are maintained.
     * @param cut_points_log Log of last points selected.
     * @param best_chr The best chromosome of the previous population
     * @return A new population of reduced chromosomes.
     */
    private void reduction(int[] cut_points_log, boolean[] best_chr){

    	ArrayList<RankingPoint> candidatePoints = new ArrayList<RankingPoint>(cut_points_log.length);
    	List<RankingPoint> newPoints = new ArrayList<RankingPoint>(cut_points_log.length);
		int reduced_size = (int) ((1 - pReduction) * cut_points_log.length);
    	
		// Maintain the best chromosome's points and the rest ones are used to form a ranking
    	for(int i = 0; i < cut_points_log.length; i++){
    		RankingPoint point = new RankingPoint(i, cut_points_log[i]);
    		if(best_chr[i])
    			newPoints.add(point);
    		else
    			candidatePoints.add(point);
    	}
		
    	// Select the best ranked points to complete the reduced chromosome
		int rest_size = reduced_size - newPoints.size();
		if(rest_size > 0) { 
			
			if(candidatePoints.size() > rest_size) {
				// Order the points by ranking (descending)
				Collections.sort(candidatePoints, new ComparePointsByRank());
				int pivot = rest_size - 1;
				int last_rank = candidatePoints.get(pivot).getRank();
				
				if (last_rank != candidatePoints.get(pivot + 1).getRank()) {
					// We add all best ranked candidates until completing the reduced chrom
					newPoints.addAll(candidatePoints.subList(0, rest_size));	
				} else {
					// We have to select the last ranked randomly
					int first_pos = 0;
					for(int i = pivot; i >= 0; i--) {
						if(last_rank != candidatePoints.get(i).getRank()){
							first_pos = i + 1;
							break;
						}
					}
					
					int last_pos = candidatePoints.size();
					for(int i = pivot; i < candidatePoints.size(); i++) {
						if(last_rank != candidatePoints.get(i).getRank()){
							last_pos = i;
							break;
						}
					}
					
					List<RankingPoint> lastRanked = candidatePoints.subList(first_pos, last_pos);
					Random r = new Random(seed);
					
					// We remove the last ranked elements until we achieve the size
					while(lastRanked.size() + first_pos > rest_size)
						lastRanked.remove(r.nextInt(lastRanked.size()));
					
					newPoints.addAll(candidatePoints.subList(0, first_pos));
					newPoints.addAll(lastRanked);
				}
			} else {
				// All candidate points fit in the new chromosome
				newPoints.addAll(candidatePoints);	
			}
			
		} else {
			System.out.println("Limit of reduction already reached");
		}
		
    	// Order the points by position (id)
    	Collections.sort(newPoints, new ComparePointsByID());
    	
    	// Reduce the actual matrix of cut points using the most selected points' positions
    	int index_points = 0;
    	int index_att = 0;
    	float[][] new_matrix = new float[cut_points.length][];
		for(int i = 0; (i < cut_points.length) && (index_points < newPoints.size()); i++) {
			if(cut_points[i] != null) {
				List<Float> lp = new ArrayList<Float>();
				while(newPoints.get(index_points).id < index_att + cut_points[i].length){
					lp.add(cut_points[i][newPoints.get(index_points).id - index_att]);
					if(++index_points >= newPoints.size()){ break; }
				}

				new_matrix[i] = new float[lp.size()];
				for(int j = 0; j < lp.size(); j++) {
					new_matrix[i][j] = lp.get(j);
				}

				index_att += cut_points[i].length;
			}				
		}
		cut_points = new_matrix;
		
		// Reduce the size of the chromosomes according to the number of points
		n_cut_points = newPoints.size();
		
		for(int i = 0; i < population.size(); i++) {
			boolean[] old_chr = population.get(i).getIndividual();
			boolean[] new_chr = new boolean[n_cut_points];
			for(int j = 0; j < newPoints.size(); j++){
				new_chr[j] = old_chr[newPoints.get(j).id];
			}
			population.set(i, new Chromosome(new_chr));
		}   	
    	
    	//evalPopulation();
    }
    
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

