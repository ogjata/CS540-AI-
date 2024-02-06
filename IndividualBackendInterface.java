
public interface IndividualBackendInterface <T> {
  
  
  /**
   * Reads the file of car list and inserts to the Red-Black Tree
   * 
   */
  public void readFile();
  
  

  /**
   * Searches the car by its feature(brand, year, model, price and mileage)
   * 
   * @return returns the car that is being searched
   */
  public T search(String feature);
  
  
  /**
   * inserts the car into the Tree
   */
  public void insert(T car);
  
  
  /**
   * sets the car's brand
   */
  public void setBrand(String brand);
  
  
  /**
   * sets the car's model
   */
  public void setModel(String model);
  
  
  /**
   * sets the car's year
   */
  public void setYear(int year);
  
  
  /**
   * sets the car's mileage
   */
  public void setMileage(int mileage);
  
  
  
  /**
   * returns the list of Cars by milage. It will return by the variable inOrder (starting from low mileage or high mileage)
   */
  public T listOfCars(String inOrder);
  
  
  /**
   * returns the list of Cars by specific mileage threshold. 
   * It will return by the variable inOrder (starting from low mileage or high mileage)
   */
  public T listOfCars(String inOrder, int threshold);
  
  
  
}

