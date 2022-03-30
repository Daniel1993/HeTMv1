//import ZipfGenerator;

public class Main {

	public static void main(String[] args)
	{
		if (args.length != 3) {
			System.out.println("Usage: \n\tjava Main NB_ITEMS PARAM SIZE");
			System.exit(-1);
		}

		Long nb_items = Long.parseLong(args[0]);
		Double  param = Double.parseDouble(args[1]);
		Integer  size = Integer.parseInt(args[2]);

		ZipfianGenerator gen = new ZipfianGenerator(nb_items, param);
		for (int i = 0; i < size; ++i) {
			System.out.println(gen.nextValue());
		}
	}
}
