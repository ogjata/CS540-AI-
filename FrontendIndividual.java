public interface FrontendIndividual{

	/**
	 * constructor that accepts backend instantiation and scanner
	 */
	public void FrontendIndividual(Backend backend, Scanner scanner);

	/**
	 * main menue loop
	 */
	public void mainMenue();

	/**
	 * command method for file specification
	 */
	public void loadData();

	/**
	 * command to statistics about the dataset
	 */
	public void showStats();

	/**
	 *  command that asks the user for two participants, then lists
	 *  the closest connection between the two, including all intermediary friends
	 */
	public void showConnection();

	/**
	 * method for exit command
	 */
	public void exit();
}

