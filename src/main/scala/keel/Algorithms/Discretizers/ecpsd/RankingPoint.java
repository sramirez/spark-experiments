package keel.Algorithms.Discretizers.ecpsd;

import java.io.Serializable;
import java.util.Comparator;

public class RankingPoint implements Comparable, Serializable {
	 
	  public int id;
	  public int rank;
	 
	  public RankingPoint (int id, int rank) {
	    this.id = id;
	    this.rank = rank;
	  }
	 
	  public int getID() { 
		  return id; 
	  }
	  
	  public int getRank () { 
		  return rank; 
	  }
	 
	  public void setRank (int newRank) { 
		  this.rank = newRank;  
	  }
	  public String toString() {
		  return ("ID; " + id + ", RANK: " + rank);
	  }
	 
	  public boolean equals (Object o) {
		  if (o == null)
			  return false;
		  if (o == this)
			  return true;
		  if (!(o instanceof RankingPoint))
			  return false;
		  
		  RankingPoint p = (RankingPoint) o;
		  
		  return this.id != p.id;
	  }
	  
	  public int hashCode() {
		  int result = 17;
		  
		  result = 31 * result + id;
		  //result = 31 * result + (int) (Double.doubleToLongBits(proportion)^((Double.doubleToLongBits(proportion) >>> 32)));
		  
		  return result;
	  }
	  
	  public int compareTo (Object o) {
		  RankingPoint other = (RankingPoint) o;
		  return new Integer(this.rank).compareTo(other.rank);
	  }
	  
}

class CompareByID implements Comparator<RankingPoint> {
    @Override
    public int compare(RankingPoint o1, RankingPoint o2) {
    	return new Integer(o1.id).compareTo(o2.id);
    }
}

class CompareByRank implements Comparator<RankingPoint> {
    @Override
    public int compare(RankingPoint o1, RankingPoint o2) {
    	return new Integer(o1.rank).compareTo(o2.rank);
    }
}
