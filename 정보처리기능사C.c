from numpy import character

class path{
    public static void main(String[] args){
        String t1 = "4-1a2";
        String t2 = "5-523";

        int res = 0;

        for (int i =0; i<5; i++{
            try{
                res += character.getNumericValue(t2.charAt(Character.getNumericValue(t1.charAt(i))));
            } catch (Exception e) {
                res -= 1;
            } finally {
                res += 1;
            }
        }
        System.out.printf("%d", res);)
    }
}

