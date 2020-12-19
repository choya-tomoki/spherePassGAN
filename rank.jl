#=
rank:
- Julia version: 1.1
- Author: choya
- Date: 2020-01-28
=#
const train_path = "data/train.txt"
const gen_path = "gen_passwords8_4.txt"

train_pass = open(train_path, "r") do fp1
                                    readlines(fp1)
                                 end

#gen_pass = open(gen_path, "r") do fp2
#                               readlines(fp2)
#                               end

len_train_pass = length(train_pass)
t_dict = Dict()
for t in train_pass
    check = haskey(t_dict, t)
    if check
        t_dict[t] = t_dict[t] + 1
    else
        t_dict[t] = 1
    end
end


filter!(p->(last(p) > 5000), t_dict)
print("[info] finish filter\n")
array = collect(t_dict)
print("[info] finish convert array to dict\n")
sort!(array, by=x->last(x))
for i=1:length(array)
    print(array[i][1], "    ", array[i][2], "     ", array[i][2]/len_train_pass*100, " %\n")
end
