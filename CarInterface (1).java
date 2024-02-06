/**
 * This is the interface of a type that will allow us to create and access cars
 * and their attributes
 */
public interface CarInterface extends Comparable<CarInterface> {
    /*private String brand;
    private String model;
    private int year;
    private int price;
    private double mileage;*/

    /**
     * Constructs an empty Car, assigns null or 0 values to attributes
     */
    /* public void Car() {
        this.brand = null;
        this.model = null;
        this.year = 0;
        this.price = 0;
        this.mileage = 0;
    } */

    /**
     * Constructs a Car with brand, model, year, price, and mileage attributes
     *
     * @param brand   string defining car's brand
     * @param model   string defining car's model
     * @param year    int defining car's year
     * @param price   double defining car's price
     * @param mileage string defining car's mileage
     */

   /*  public Car(String brand, String model, int year, double price, int mileage) {
        this.brand = brand;
        this.model = model;
        this.year = year;
        this.price = price;
        this.mileage = mileage;
    }
    */


    /**
     * Getter for Car object's brand
     *
     * @return brand of car
     */

    public String getBrand();

    /**
     * Setter for Car object's brand
     *
     * @param brand - new brand attribute
     */
    public void setBrand(String brand);

    /**
     * Getter for Car object's model
     *
     * @return model of car
     */
    public String getModel();

    /**
     * Setter for Car object's model
     *
     * @param model - new model attribute
     */
    public void setModel(String model);

    /**
     * Getter for Car object's year
     *
     * @return year of car
     */
    public int getYear();

    /**
     * Setter for Car object's year
     *
     * @param year - new year attribute
     */
    public void setYear(int year);

    /**
     * Getter for Car object's price
     *
     * @return price of car
     */
    public int getPrice();

    /**
     * Setter for Car object's price
     *
     * @param price - new price attribute
     */
    public void setPrice(int price);

    /**
     * Getter for Car object's mileage
     *
     * @return mileage of car
     */
    public double getMileage();

    /**
     * Setter for Car object's mileage
     *
     * @param mileage - new mileage attribute
     */
    public void setMileage(double mileage);

    int compareTo(Car car);

    public int compareTo(CarInterface car);


    public String toString();
}
