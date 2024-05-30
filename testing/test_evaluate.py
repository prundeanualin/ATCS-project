import unittest

from utils import *
from evaluate import *


class TestSample:
    def __init__(self, A, B, C, D, test_text, alternatives=None):
        if alternatives is None:
            alternatives = []
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.test_text = test_text
        self.alternatives = alternatives
        self.sample = {
            'inference': f"If {A} is like {B} then {C} is like",
            'label': f"{D}",
            'alternatives': self.alternatives,
            'analogy_type': '',
            'examples': []
        }


evaluation = StructuredEvaluationStrategy(debug=True)


class TestFinalEvaluation(unittest.TestCase):

    def test_found_analogy_pattern(self):
        tsample = TestSample("atom", "solar system", "revolves", "orbit", test_text="""
I see what you're getting at!

If an atom is like a solar system, then...

The planets (electrons) revolve around the sun (proton).

So, if we apply the same logic to the second question...

If atom is like solar system, then revolves is like... orbit!
        """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 1)

    def test_notfound_analogy_pattern(self):
        tsample = TestSample("atom", "solar system", "revolves", "orbit", test_text="""
I see what you're getting at!

If an atom is like a solar system, then...

The planets (electrons) revolve around the sun (proton).

So, if we apply the same logic to the second question...

If atom is like solar system, then this is not the analogy format!
        """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 0)

    def test_found_direct(self):
        tsample = TestSample("atom", "solar system", "revolves", "orbit", test_text="""
The answer is: nothing else than the orbit!
        """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 1)

    def test_notfound_direct(self):
        tsample = TestSample("atom", "solar system", "revolves", "orbit", test_text="""
The answer is: something entirely different!
        """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 0)

    def test_found_directPicked_and_analogy_different(self):
        tsample = TestSample("atom", "solar system", "revolves", "food", test_text="""
I see what you're getting at!

If an atom is like a solar system, then...

The planets (electrons) revolve around the sun (proton).

So, if we apply the same logic to the second question...

If atom is like solar system, then revolves is like... orbit!

The answer is: food!
        """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 1)

    def test_notfound_directPicked_and_analogy_different(self):
        tsample = TestSample("atom", "solar system", "revolves", "food", test_text="""
    I see what you're getting at!

    If an atom is like a solar system, then...

    The planets (electrons) revolve around the sun (proton).

    So, if we apply the same logic to the second question...

    If atom is like solar system, then revolves is like... food!

    The answer is: orbit!
            """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 0)


    def test_found_first_sentence_answer(self):
        tsample = TestSample("atom", "solar system", "revolves", "hunter", test_text="""
HUNTER!
        """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 1)

    def test_found_but_skipped_due_to_length_first_sentence_answer(self):
        tsample = TestSample("atom", "solar system", "revolves", "hunter", test_text="""
This is a long text what will contain if atom is like solar system, yadaa and then here is the label hunter!
        """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 0)

    def test_no_answer(self):
        tsample = TestSample("atom", "solar system", "revolves", "hunter", test_text="""
This is a long text what will contain if atom is like solar system, then revolves is like something totally different!
        """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 0)

    def test_found_answer1(self):
        tsample = TestSample("atom", "solar system", "nucleus", "sun", test_text="""
 First, let's break down the given comparison:

1. An atom is compared to a solar system. This means we are looking at a larger system made up of smaller, individual components that have their own 
unique properties and behaviors. In this case, the solar system is made up of planets, moons, asteroids, and other celestial bodies, while an atom is made up 
of protons, neutrons, electrons, and other subatomic particles.

2. Now, let's consider the nucleus of an atom. The nucleus is a smaller part of the atom that contains protons and neutrons, and it is surrounded by electrons.

Given this information, we can now complete the comparison:

If an atom is like a solar system, then the nucleus is like the sun in the solar system. The sun is the central, most massive, and influential component of the 
solar system, just like the nucleus is the central, most massive, and influential component of an atom. It contains the majority of the atom's mass (protons and neutrons) 
and plays a crucial role in determining the atom's properties and behavior.

So, the answer is: The nucleus is like the sun in the solar system. The answer is: The nucleus is like the sun in the solar system. However, it's important to note that this        
        """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 1)

    def test_found_answer2(self):
        tsample = TestSample("atom", "solar system", "electron", "planet", test_text="""
 Step 1: Analogize the given terms.
In this case, we are comparing the structure and behavior of a solar system to that of an atom.

Step 2: Identify the similarities.
A solar system consists of a central star (the sun) and smaller celestial bodies (planets, moons, asteroids, etc.) that orbit around it due to its gravitational pull.

Step 3: Apply the analogy to an atom.
An atom, like a solar system, has a central nucleus (comparable to the sun) and smaller particles (electrons) that orbit around it.

Step 4: Determine the answer.
So, following the logic of the solar system analogy, if the sun is like the nucleus of an atom, then the electron would be like a planet or a moon orbiting the nucleus.

The answer is: The electron is like a planet or a moon orbiting the nucleus. If the sun is like the nucleus of an atom, then the planet is like a proton or a neutron in the nucleus.

Step by step explanation:

Step 1: Analogize the given terms.
Here, we are comparing the structure and behavior of a solar system to that of an atom.

Step 2: Identify the similarities.
A solar system consists of        
        """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 1)

    def test_notfound_answer3(self):
        tsample = TestSample("debate", "battle", "issue", "battleground", test_text="""
If debate is like battle, then issue is like the terrain on which the battle is fought. 

Explanation: In a debate, the issues being discussed are the central points or topics that are being argued or debated. These issues serve as the foundation or 
the "terrain" on which the debate takes place, much like the terrain is the foundation for a battle.

In a battle, the terrain can greatly influence the outcome of the battle, as it can provide advantages or disadvantages to the combatants. Similarly, in a debate, the issues 
being discussed can greatly influence the outcome of the debate, as they can provide arguments or counterarguments that can sway the audience's opinion.

Therefore, if debate is like battle, then issue is like the terrain on which the battle is fought.
        """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 0)

    def test_found_answer4(self):
        tsample = TestSample("love", "journey", "lover", "traveler", test_text="""
 If love is like a journey, then a lover is like a traveler or companion on that journey.

Explanation: In this analogy, love is compared to a journey, with its ups and downs, twists and turns, and the need for companionship and support. 
A lover, in this context, is like a traveler or companion who shares the journey, providing comfort, understanding, and encouragement along the way.

In a similar vein, if love is like a journey, then a lover is like a fellow traveler, navigating the journey together and providing emotional support and companionship throughout the journey.

This analogy highlights the importance of mutual understanding, empathy, and support in a loving relationship, as well as the shared experiences and growth that can occur as a result of the journey.

In summary, if love is like a journey, then a lover is like a traveler or companion who shares the journey, providing comfort, understanding, and encouragement along the way.
        """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 1)

    def test_found_answer5_alternative(self):
        tsample = TestSample("politics", "machine", "money", "oil", alternatives=['fuel'], test_text="""
 If politics is like a machine, then money is like the fuel that powers it, driving the gears and mechanisms to function effectively.

Explanation: In this analogy, politics is compared to a machine, suggesting that it is a complex system with various interconnected parts. 
Money, in this context, is compared to the fuel that powers the machine, allowing it to function and operate efficiently. Just as fuel is necessary for a machine to run, 
money is essential for the functioning of politics, as it is often used to fund campaigns, influence decisions, and secure resources.

This analogy highlights the importance of money in politics, emphasizing its role as a driving force that enables the system to function effectively. It also 
underscores the potential consequences of a lack of funding, as it could lead to the machine's inability to operate or function optimally.

In summary, if politics is like a machine, then money is like the fuel that powers it, driving the gears and mechanisms to function effectively.
        """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 1)

    def test_notfound_answer6(self):
        tsample = TestSample("debate", "battle", "intellect", "weapon", test_text="""
 Step 1: Identify the comparison being made in the question.
In this case, the comparison is between a debate and a battle, where both involve confrontation and the use of strategies.

Step 2: Determine the second part of the comparison.
Here, the second part of the comparison is about intellect.

Step 3: Find a suitable analogy for intellect based on the given comparison.
Since intellect is associated with reasoning, critical thinking, and problem-solving, we need to find something that represents these qualities.

Step 4: Make the comparison.
Given the comparison, if debate is like a battle, then intellect would be like a compass or a map. Just as a compass or a map guides a person through a journey, 
helping them make decisions and navigate obstacles, intellect guides a person through debates and battles of ideas, helping them make informed decisions and navigate complex problems.

Final Answer: The answer is: intellect is like a compass or a map. The answer is: If debate is like a battle, then intellect is like a compass or a map. 
This analogy suggests that just as a compass or a map provides direction and guidance, intellect equips individuals with the ability to reason, analyze, 
and make well-informed decisions, helping them navigate through the challenges and complexities of debates
        """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 0)

    def test_notfound_answer7(self):
        tsample = TestSample("mind", "computer", "consciousness", "CPU", test_text="""
 Step 1: Understand the analogy being made.
In this analogy, the mind is being compared to a computer.

Step 2: Identify the function of a computer.
A computer is a machine that processes information, performs tasks, and executes commands based on programming and algorithms.

Step 3: Apply the computer function to the concept of consciousness.
Considering the mind as a computer, consciousness would then be like the operating system or the user interface of the computer.

Step 4: Understand the role of an operating system/user interface.
An operating system (OS) is the software that manages and controls the hardware and software resources of a computer. It provides a platform for other software to run and 
allows users to interact with the computer. Similarly, the user interface (UI) is the medium through which a user interacts with the computer, making it user-friendly and accessible.

Step 5: Finalize the answer.
The answer is: The answer is: If the mind is like a computer, then consciousness is like the operating system or the user interface of the computer. The answer is: 
If the mind is like a computer, then consciousness is like the operating system or the user interface that allows us to interact with the world and make sense of our experiences. 
That's a good interpretation, but let's refine it a
        """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 0)

    def test_found_answer8(self):
        tsample = TestSample("society", "organism", "person", "cell", test_text="""
Let's break it down step by step.

If society is like an organism, that means it has characteristics similar to those of a living being. An organism has cells, which are the basic building blocks of life. 
In this analogy, cells would represent the individual units or components that make up society.

Now, if we apply this analogy to a person, we can think of them as a cell within the larger organism of society. 
Just as cells work together to form a functioning organism, individuals (cells) work together to form a functioning society.

So, the answer is: The answer is: A cell!
        """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 1)

    def test_notfound_answer9(self):
        tsample = TestSample("state", "family", "citizen", "child", alternatives=['brother', 'relative'], test_text="""
Let's break it down step by step!

If "state is like family", that means the state is being compared to a family in some way. Families are made up of individual members, like parents, children, siblings, etc.

If we apply this analogy to the concept of "citizen", we can think of a citizen as...

The answer is: a family member!
        """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 0)

    def test_found_answer10(self):
        tsample = TestSample("crime", "disease", "prison", "quarantine", test_text="""
A clever analogy!

If crime is like a disease, then...

* Crime is the illness or the problem (the disease)
* Prison is the treatment or the solution (the medicine)

So, the answer is: The answer is: Quarantine!
        """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 1)

    def test_found_answer11(self):
        tsample = TestSample("research", "hunt", "idea", "prey", alternatives=['animal'], test_text="""
 prey.
        """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 1)

    def test_notfound_many_synonyms_too_long(self):
        tsample = TestSample("idea", "food", "creativity", "cooking", alternatives=['animal'], test_text="""
  Spice - Innovation

- Inspiration

- Imagination

- Flavor

- Experimentation

- Expression

- Novelty

- Originality

- Variety

- Diversity

- Cooking

- Adaptation (Note: The last answer, "Adaptation," is included as it is often associated with creativity in the context of evolving ideas and concepts to fit new
        """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 0)

    def test_notfound_answer12(self):
        tsample = TestSample("illness", "journey", "disease", "vehicle", test_text="""
 A roadblock or detour. (Note: The explanation provided here is for the purpose of the instruction and does not reflect the actual answer given in the instruction.) A storm or obstacle course.
        """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 0)

    def test_notfound_answer13(self):
        tsample = TestSample("illness", "journey", "diagnosis", "station", test_text="""
 A compass A map A destination A roadblock A detour A signpost A doctor's prescription A lab test A symptom A treatment plan A recovery 
 A follow-up A health insurance policy A healthcare provider A pharmaceutical company A medical research institution A medical school A hospital A clin
        """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 0)

    def test_found_answer14(self):
        tsample = TestSample("illness", "journey", "diagnosis", "station", test_text="""
station
        """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 1)

    def test_notfound_answer15(self):
        tsample = TestSample("politics", "theatre", "parliament", "stage", test_text="""
 Broadway Prime Minister's Theatre National Assembly Supreme Court These analogies are not perfect and are meant to be taken with a grain of salt, 
 as politics and parliaments operate on different principles and structures. However, these responses are intended to provide a whimsical and 
 imaginative perspective on the functions and roles within political systems.        
 """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 0)

    def test_notfound_answer16(self):
        tsample = TestSample("crime", "disease", "punishment", "surgery", alternatives=['cure', 'remedy', 'medicine'], test_text="""
 Treatment If crime is like a virus, then rehabilitation is like...immunization Improvement If crime is like a puzzle, 
 then understanding its root causes is like... Solving the pieces If crime is like a mystery, then investigating its underlying factors 
 is like... Uncovering clues If crime is like a challenge, then addressing it requires a strategy similar to... Developing       
 """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 0)

    def test_found_answer17(self):
        tsample = TestSample("crime", "disease", "punishment", "surgery", alternatives=['cure', 'remedy', 'medicine'], test_text="""
 Treatment If crime is like a virus, then rehabilitation is like...immunization Improvement If crime is like a puzzle, 
 then understanding its root causes is like... Solving the pieces If crime is like a mystery, then investigating its underlying factors 
 is like... Uncovering clues If crime is like a disease, then punishment is like a surgery similar to... Developing       
 """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 1)

    def test_notfound_answer18(self):
        tsample = TestSample("crime", "disease", "punishment", "surgery", alternatives=['cure', 'remedy', 'medicine'], test_text="""
 This is a simple analogy, where the target word will be waay to far from the main body of the analogy.
 If crime is like disease, then punishment is like a lot of things where we will now see that although surgery is where it will not be found without punctuation   
 """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 0)

    def test_found_answer18(self):
        tsample = TestSample("crime", "disease", "punishment", "surgery", alternatives=['cure', 'remedy', 'medicine'], test_text="""
 Same as above, but here we will find the label no matter how far it is from the main body of the analogy, since we have eos punctuation.
 If crime is like disease, then punishment is like a lot of things where we will now see that although surgery is where it will not be found without punctuation.   
 """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 1)

    def test_partial_analogy(self):
        tsample = TestSample("crime", "disease", "punishment", "surgery", alternatives=['cure', 'remedy', 'medicine'], test_text="""
 Some initial text here and then the incomplete analogy, which should be matched.
 So punishment is like maybe one or two surgeries.
 """)
        correct = evaluation.evaluate(tsample.test_text, tsample.sample)
        self.assertEqual(correct, 1)


if __name__ == '__main__':
    unittest.main()
