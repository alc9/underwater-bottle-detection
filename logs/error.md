After reading the code I think I know what the issue was - if you ran with --no_save giving a default true but final_epoch condition was only met at the final epoch you set then the model would only save once at the --epoch num point
if save_period not set then only save @ best_fitness
best_fitness only saves if not nosave or final_epoch and not evolve
final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
default early stopping patience is 100 - possible_stop returns boolean
 # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
