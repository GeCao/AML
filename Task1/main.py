from Task1.src.CoreManagement import CoreComponent


if __name__ == "__main__":
    core_managemnet = CoreComponent(model='lasso', device='cuda')
    core_managemnet.initialization()
    core_managemnet.run()
    core_managemnet.kill()