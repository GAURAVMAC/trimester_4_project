# def winner(user_score, system_score): return f'Result: User Won with {abs(user_score-system_score)} Runs' if user_score > system_score else f'Result: Tie' if user_score==system_score else f'Result: System Won with {abs(user_score-system_score)} Runs'

# user_score = 9
# system_score = 9
# print(winner(user_score, system_score))

class Score():
    def __init__(self, user_score=0, system_score=0, over=[], system_array=[]):
        self.user_score = user_score
        self.system_score = system_array
        self.over = over
        self.system_array = system_array

    def __str__(self): return f'Result: User Won with {abs(self.user_score-self.system_score)} Runs' if self.user_score > self.system_score else f'Result: Tie' if self.user_score==self.system_score else f'Result: System Won with {abs(self.user_score-self.system_score)} Runs'


score = Score()

score.user_score = 3
score.system_score = 2

# print(score.user_score)
# print(score.system_score)
# print(score)

print(len(score))
print()