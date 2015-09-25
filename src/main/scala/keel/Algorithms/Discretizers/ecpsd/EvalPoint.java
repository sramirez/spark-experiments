package keel.Algorithms.Discretizers.ecpsd;

import java.io.Serializable;
import java.util.Comparator;

public class EvalPoint implements Comparable, Serializable {
	 
	  public int n_err;
	  public int ninst;
	  
	  public EvalPoint(int ne, int ni) {
		// TODO Auto-generated constructor stub
		  n_err = ne;
		  ninst = ni;			  
	  }

	  public void add(EvalPoint other) {
		  n_err += other.n_err;
		  ninst += other.ninst;
		  
	  }
	@Override
	public int compareTo(Object arg0) {
		// TODO Auto-generated method stub
		return 0;
	}
}
