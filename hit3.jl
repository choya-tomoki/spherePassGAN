#=
hit3:
- Julia version: 1.1
- Author: choya
- Date: 2020-01-24
=#
const train_path = "data/train.txt"
const test_path = "data/test_unique_sorted.txt"
const gen_path = "gen_sample/gen_passwords8_2.txt"
const hit_passwords = "hit_passwords.txt"

test_pass = open(test_path, "r") do fp2
                                    readlines(fp2)
                                 end

gen_pass = open(gen_path, "r") do fp3
                               readlines(fp3)
                               end

hit_pass = Set(open(hit_passwords, "r") do fp4
                                readlines(fp4)
                                end)

const original_genpass_len = length(gen_pass)

#gen_pass = filter(x -> !(x in train_pass), gen_pass)
#print("filter genpass done\n")

hit_count = 0
const test_pass_len = length(test_pass)

print("[info] start sortingn\n")
sort!(gen_pass)
print("[info] finish sorting\n")

for (n, tp) in enumerate(test_pass)
    global hit_count, hit_pass
    if n % 10000 == 0
        print("hit  :  ", n, " / ", test_pass_len, "\n")
    end

    low = 1
    high = original_genpass_len
    flag = false
    while(low <= high)
        mid = div(low + high, 2)
        guess = gen_pass[mid]
        if guess == tp
            flag = true
            break
        elseif guess > tp
            high = mid - 1
        else
            low = mid + 1
        end
    end
    if flag == true
       hit_count += 1
       push!(hit_pass, tp)
    end
end

hit_par = hit_count * 100 / test_pass_len
print("Passwords Generated : ", original_genpass_len,  "  matched : ", hit_count, "(", hit_par, " %)", "\n")
print(test_pass_len, "   unique test samples\n")

fw = open(hit_passwords, "w")
for h in hit_pass
    println(fw, h)
end
close(fw)

print("total hit : ", length(hit_pass), "    total matched : ", length(hit_pass)*100/test_pass_len)