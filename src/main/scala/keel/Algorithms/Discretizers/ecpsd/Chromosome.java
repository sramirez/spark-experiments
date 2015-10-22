package keel.Algorithms.Discretizers.ecpsd;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
/**
 * <p>Title: CHC_RuleBase </p>
 *
 * <p>Description: Chromosome that represents a rule base used in the CHC algorithm </p>
 *
 * <p>Company: KEEL </p>
 *
 * @author Written by Victoria Lopez (University of Granada) 30/04/2011
 * @author Modified by Sergio Ramirez (University of Granada) 04/05/2014
 * @version 1.5
 * @since JDK1.5
 */

public class Chromosome implements Comparable, Serializable {
	private boolean [] individual; // Boolean array selecting cutpoints from a list of cutpoints
	private boolean n_e; // Indicates whether this chromosome has been evaluated or not
	float fitness;// Fitness associated to the cut points represented by the boolean array
	int n_cutpoints; // Fitness associated to the dataset, it indicates the number of cutpoints selected
	int inconsistencies;
	float perc_err;
	
	/**
     * Default constructor
     */
    public Chromosome () {
    }
    
    /**
     * Creates a CHC chromosome from another chromosome (copies a chromosome)
     * 
     * @param orig	Original chromosome that is going to be copied
     */
    public Chromosome (Chromosome orig) {
    	individual = new boolean [orig.individual.length];
    	
    	for (int i=0; i<orig.individual.length; i++) {
    		individual[i] = orig.individual[i];
    	}
    	
    	n_e = orig.n_e;
    	fitness = orig.fitness;
    	n_cutpoints = orig.n_cutpoints;
    }
    
    /**
     * Creates a random CHC_Chromosome of specified size
     * 
     * @param size	Size of the new chromosome 
     */
    public Chromosome (int size) {
    	
    	individual = new boolean [size];
    	
    	Random rnd = new Random();
    	for (int i=0; i<size; i++) {
    		float u = rnd.nextFloat();
			if (u < 0.5) {
				individual[i] = false;
			}
			else {
				individual[i] = true;
			}
    	}
    	
    	n_e = true;
    	fitness = .0f;
    	n_cutpoints = size;
    }

    /**
     * Creates a CHC_Chromosome of specified size with all its elements set to the specified value
     * 
     * @param size	Size of the new chromosome
     * @param value	Value that all elements of the chromosome are going to have 
     */
    public Chromosome (int size, boolean value) {
    	individual = new boolean [size];
    	
    	for (int i=0; i<size; i++) {
    		individual[i] = value;
    	}
    	
    	n_e = true;
    	fitness = .0f;
    	n_cutpoints = size;
    }
    
    /**
     * Creates a CHC chromosome from a boolean array representing a chromosome
     * 
     * @param data	boolean array representing a chromosome
     */
    public Chromosome (boolean data[]) {
    	individual = new boolean [data.length];
    	
    	for (int i=0; i<data.length; i++) {
    		individual[i] = data[i];
    	}
    	
    	n_e = true;
    	fitness = .0f;
    	n_cutpoints = data.length;
    }
    
    /**
     * Creates a CHC chromosome from another chromosome using the CHC diverge procedure
     * 
     * @param orig	Best chromosome of the population that is going to be used to create another chromosome
     * @param r	R factor of diverge
     */
    public Chromosome (Chromosome orig, float r) {
    	individual = new boolean [orig.individual.length];
    	
    	Random rnd = new Random();
    	for (int i=0; i<orig.individual.length; i++) {
    		if (rnd.nextFloat() < r) {
    			individual[i] = !orig.individual[i];
    		}
    		else {
    			individual[i] = orig.individual[i];
    		}
    	}
    	
    	n_e = true;
    	fitness = .0f;
    	n_cutpoints = orig.n_cutpoints;
    }

    /**
     * Creates a CHC chromosome from another chromosome using the CHC diverge procedure
     * 
     * @param orig	Best chromosome of the population that is going to be used to create another chromosome
     * @param r	R factor of diverge
     */
    public Chromosome (Chromosome orig, float r, float prob0to1Div) {
    	individual = new boolean [orig.individual.length];
    	
    	Random rnd = new Random();
    	for (int i=0; i<orig.individual.length; i++) {
    		if (rnd.nextFloat() < r) {
    			if (rnd.nextFloat() < prob0to1Div) {
    				individual[i] = true;
    			} else {
    				individual[i] = false;
    			}
    		}
    		else {
    			individual[i] = orig.individual[i];
    		}
    	}
    	
    	n_e = true;
    	fitness = .0f;
    	n_cutpoints = individual.length;
    }
    
    /**
     * Checks if the current chromosome has been evaluated or not.
     * 
     * @return true, if this chromosome was evaluated previously;
     * false, if it has never been evaluated
     */
    public boolean not_eval() {
        return n_e;
    }
 
    /**
     * Evaluates this chromosome, computing the fitness of the chromosome
     * 
     * @param dataset	Training dataset used in this algorithm
     * @param all_cut_points	Proposed cut points that are selected by the CHC chromosome
     * @param alpha	Coefficient for the number of cut points importance
     * @param beta	Coefficient for the number of inconsistencies importance
     */
    public void evaluate (weka.core.Instances base, float[][] dataset, float [][] cut_points, float alpha, float beta) {
    	
    	weka.core.Instances datatrain = new weka.core.Instances(base);
    	int nInputs = dataset[0].length - 1;
    	
    	// Obtain the cut points for this chromosome
    	ArrayList<Float[]> selected_points = 
    			new ArrayList<Float[]>(cut_points.length);
    	int acc = 0, n_selected_cut_points = 0;
    	
    	for (int i=0; i < cut_points.length; i++) {
    		ArrayList<Float> arr = new ArrayList<Float>();
    		for (int j = 0; j < cut_points[i].length; j++) {
    			if (individual[acc + j]){
    				arr.add(cut_points[i][j]);
    				n_selected_cut_points++;
    			} 
			}
    		Float[] aux = new Float[arr.size()];
    		selected_points.add(arr.toArray(aux));
    		acc += cut_points[i].length;
    	}
    	
	    /*Instance adaptation to the WEKA format*/
	    int j;
    	for (int i=0; i < dataset.length; i++) {
    		float [] sample = dataset[i];
    		double[] tmp = new double[sample.length];    		
    		for (j=0; j < nInputs; j++) 
    			tmp[j] = discretize (sample[j], selected_points.get(j));
    		
    		tmp[j] = sample[nInputs]; // the class
    		DenseInstance inst = new DenseInstance(1.0, tmp);
    		
    		datatrain.add(inst);
    	}    	
    	
    	/* Use unpruned C4.5 to detect errors
    	 * Second type of evaluator in precision: C45 classifier
    	 * c45er is the error counter
    	 * */
    	//long t_ini = System.currentTimeMillis();
    	
    	/*double c45er = 0;
	    J48 baseTree = new J48();		
	    
	    try {
	    	baseTree.buildClassifier(datatrain);	    		
	    } catch (Exception ex) {
	    	ex.printStackTrace();
	    }
	    
	    for (int i=0; i < datatrain.numInstances(); i++) {
	    	try {
	    		if ((int)baseTree.classifyInstance(datatrain.instance(i)) 
	    				!= dataset[i][nInputs]) {
	    			c45er++;
	    		}
		    } catch (Exception ex) {
		    	ex.printStackTrace();
		    }
	    }*/
	    
    	/* Use simple Naive Bayes to detect errors
    	 * Third type of evaluator in precision: Naive Bayes classifier
    	 * nber is the error counter
    	 * */
    	int nber = 0;
	    NaiveBayes baseBayes = new NaiveBayes();		
	    
	    try {
	    	baseBayes.buildClassifier(datatrain);	    		
	    } catch (Exception ex) {
	    	ex.printStackTrace();
	    }
	    
	    for (int i=0; i < datatrain.numInstances(); i++) {
	    	try {
	    		if ((int)baseBayes.classifyInstance(datatrain.instance(i)) 
	    				!= dataset[i][nInputs]) {
	    			nber++;
	    		}
		    } catch (Exception ex) {
		    	ex.printStackTrace();
		    }
	    }
	    
	    /*long t_fin = System.currentTimeMillis();
        long t_exec = t_fin - t_ini;
        long hours = t_exec / 3600000;
        long rest = t_exec % 3600000;
        long minutes = rest / 60000;
        rest %= 60000;
        long seconds = rest / 1000;
        rest %= 1000;
        System.out.println("Wrapper execution Time: " + hours + ":" + minutes + ":" +
                seconds + "." + rest);   */
	    float p_err = (float) nber / datatrain.numInstances();
	    //float proportion = (double) (dataset.getnData() * 2) / (double) initial_cut_points;
	    float perc_points= (float) n_selected_cut_points / individual.length;
        /* fitness = alpha * ((double) n_selected_cut_points / (double) initial_cut_points) 
        		+ beta * proportion * ((double) incons / (double) dataset.getnData()) ;*/
    	fitness = alpha * perc_points + beta * p_err;
        n_cutpoints = n_selected_cut_points;
        perc_err = p_err;
        //System.out.println(fitness);
    }
    
    /**
     * Obtains the fitness associated to this CHC_Chromosome, its fitness measure
     * 
     * @return	the fitness associated to this CHC_Chromosome
     */
    public float getFitness() {
    	return fitness;
    }
    
    /**
     * Obtains the Hamming distance between this and another chromosome
     * 
     * @param ch_b	Other chromosome that we want to compute the Hamming distance to
     * @return	the Hamming distance between this and another chromosome
     */
    public int hammingDistance (Chromosome ch_b) {
    	int i;
    	int dist = 0;
    	
    	if (individual.length != ch_b.individual.length) {
    		System.err.println("The CHC Chromosomes have different size so we cannot combine them");
    		System.exit(-1);
    	}
    	
    	for (i=0; i<individual.length; i++){
    		if (individual[i] != ch_b.individual[i]) {
    			dist++;
    		}
    	}

    	return dist;
    }
    
    
    /**
     * Obtains a new pair of CHC_chromosome from this chromosome and another chromosome, swapping half the differing bits at random
     * 
     * @param ch_b	Other chromosome that we want to use to create another chromosome
     * @return	a new pair of CHC_chromosome from this chromosome and the given chromosome
     */
    public ArrayList <Chromosome> createDescendants (Chromosome ch_b) {
    	int i, pos;
    	int different_values, n_swaps;
    	int [] different_position;
    	Chromosome descendant1 = new Chromosome();
    	Chromosome descendant2 = new Chromosome();
    	ArrayList <Chromosome> descendants;
    	
    	if (individual.length != ch_b.individual.length) {
    		System.err.println("The CHC Chromosomes have different size so we cannot combine them");
    		System.exit(-1);
    	}
    	
    	different_position = new int [individual.length];
    	
    	descendant1.individual = new boolean[individual.length];
    	descendant2.individual = new boolean[individual.length];
    	
    	different_values = 0;
    	for (i=0; i<individual.length; i++){
    		descendant1.individual[i] = individual[i];
    		descendant2.individual[i] = ch_b.individual[i];
    		
    		if (individual[i] != ch_b.individual[i]) {
    			different_position[different_values] = i;
    			different_values++;
    		}
    	}
    	
    	n_swaps = different_values/2;
    	
    	if ((different_values > 0) && (n_swaps == 0))
    		n_swaps = 1;
    	
    	Random rnd = new Random();
    	for (int j=0; j<n_swaps; j++) {
    		different_values--;
    		pos = rnd.nextInt(different_values + 1);
    		
    		boolean tmp = descendant1.individual[different_position[pos]];
    		descendant1.individual[different_position[pos]] = descendant2.individual[different_position[pos]];
    		descendant2.individual[different_position[pos]] = tmp;
    		
    		different_position[pos] = different_position[different_values];
    	}
    	
    	descendant1.n_e = true;
    	descendant2.n_e = true;
    	descendant1.fitness = .0f;
    	descendant2.fitness = .0f;
    	descendant1.n_cutpoints = individual.length;
    	descendant2.n_cutpoints = individual.length;
    	
    	descendants = new ArrayList <Chromosome> (2);
    	descendants.add(descendant1);
    	descendants.add(descendant2);

    	return descendants;
    }    
    
    /**
     * Obtains a new pair of CHC_chromosome from this chromosome and another chromosome, 
     * swapping half the differing bits at random
     * 
     * @param ch_b	Other chromosome that we want to use to create another chromosome
     * @return	a new pair of CHC_chromosome from this chromosome and the given chromosome
     */
    public ArrayList <Chromosome> createDescendants (Chromosome ch_b, float prob0to1Rec) {
    	int i;
    	Chromosome descendant1 = new Chromosome();
    	Chromosome descendant2 = new Chromosome();
    	ArrayList <Chromosome> descendants;
    	
    	if (individual.length != ch_b.individual.length) {
    		System.err.println("The CHC Chromosomes have different size so we cannot combine them");
    		System.exit(-1);
    	}
    	
    	descendant1.individual = new boolean[individual.length];
    	descendant2.individual = new boolean[individual.length];
    	
    	for (i=0; i<individual.length; i++){
    		descendant1.individual[i] = individual[i];
    		descendant2.individual[i] = ch_b.individual[i];
    		
    		Random rnd = new Random();
    		if ((individual[i] != ch_b.individual[i]) && rnd.nextFloat() < 0.5) {
    			if (descendant1.individual[i]) 
    				descendant1.individual[i] = false;
				else if (rnd.nextFloat() < prob0to1Rec) 
					descendant1.individual[i] = true;
    			
				if (descendant2.individual[i]) 
					descendant2.individual[i] = false;
				else if (rnd.nextFloat() < prob0to1Rec) 
					descendant1.individual[i] = true;
			}
    	}
    	
    	descendant1.n_e = true;
    	descendant2.n_e = true;
    	descendant1.fitness = .0f;
    	descendant2.fitness = .0f;
    	descendant1.n_cutpoints = individual.length;
    	descendant2.n_cutpoints = individual.length;
    	
    	descendants = new ArrayList <Chromosome> (2);
    	descendants.add(descendant1);
    	descendants.add(descendant2);

    	return descendants;
    }
    
    /**
     * Obtain the boolean array representing the CHC Chromosome
     * 
     * @return	boolean array selecting rules from a rule base
     */
    public boolean [] getIndividual () {
    	return individual;
    }
	
    /**
     * Obtains the discretized value of a real data considering the
     * cut points vector given and the individual information (uses binary search).
     * 
     * 
     * @param value	Real value we want to discretize according to the considered cutpoints (must be sorted in ascending order)
     * @param cp Cut points used to discretize (selected by the chromosome)
     * @return	a discrete integer value
     */
    private int discretize(float value, Float[] cp) {
    	if(cp == null) return 0; // No boundary points
    	int ipoint = Arrays.binarySearch(cp, value);
    	if(ipoint < 0) 
    		return Math.abs(ipoint) - 1;
    	else
    		return ipoint;
    }
	
    public String toString() {
    	String output = "";
		  
    	for (int i=0; i<individual.length; i++) {
    		if (individual[i]) {
    			output = output + "1 ";
    		} else {
			  	output = output + "0 ";
    		}
    	}
		  
    	return (output);
    }
    
    /**
     * Compares this object with the specified object for order, according to the fitness measure 
     * 
     * @return a negative integer, zero, or a positive integer as this object is less than, equal to, or greater than the specified object
     */
    public int compareTo (Object aThat) {
        final int BEFORE = -1;
        final int EQUAL = 0;
        final int AFTER = 1;
        
    	if (this == aThat) return EQUAL;
    	
    	final Chromosome that = (Chromosome)aThat;
    	
    	if (this.fitness < that.fitness) return BEFORE;
        if (this.fitness > that.fitness) return AFTER;
        
        if (this.n_cutpoints < that.n_cutpoints) return AFTER;
        if (this.n_cutpoints > that.n_cutpoints) return BEFORE;
        return EQUAL;
    }
    
}

